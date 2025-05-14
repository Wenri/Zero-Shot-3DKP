import json
import math
from operator import itemgetter
import os
import sys
from collections import OrderedDict, defaultdict, Counter
from contextlib import suppress, nullcontext
from pathlib import Path
from types import NoneType
from xml.etree.ElementTree import ParseError
from functools import partialmethod
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw
from einops import rearrange
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from fast_hdbscan import HDBSCAN
from tqdm import trange
from matplotlib import colors as mcolors
from data_creation.backprojection import views_from_model, RenderO3D, get_depth_point_cloud, debug_enabled
from data_creation.gpt4o import GPT4o
from data_creation.molmo import Molmo
from kpeval import KPNetIO
from kp_utils import sample_view_points


class KPNetGenerator(RenderO3D):
    COLOR_NAMES = OrderedDict(a for a in mcolors.CSS4_COLORS.items() if any(ImageColor.getrgb(a[1])))
    COLOR_MAP = {bytes.fromhex(s.lstrip('#')): v for v, s in enumerate(COLOR_NAMES.values())}
    Multimodal = Molmo
    KPIO = KPNetIO
    views_from_model = partialmethod(views_from_model)

    def __init__(self, log_dir: str | os.PathLike = Path(), expname=f'{type(Multimodal).__name__}PTS', res=512, scale=2):
        self.log_dir = Path(log_dir)
        self.gpt = GPT4o()
        self.molmo = None
        self.io = self.KPIO(self.log_dir / expname)
        self.dist = 1
        self.res = res
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis = debug_enabled()
        self.views = sample_view_points(self.dist, partition=3)
        self.proj_radius = 15
        self.kp_initialized_empty = True
        hires = (np.asarray(self.res) * self.scale).astype(np.int64)
        super(KPNetGenerator, self).__init__(self.device, res=hires.tolist())

    @torch.inference_mode()
    def get_kp_names(self, tensor_images):
        image = Image.fromarray(rearrange(tensor_images[0], 'c h w -> h w c').cpu().numpy())
        response = self.gpt.get_kplist(image)
        content = json.loads(response.choices[0].message.content)
        kp_list = [kp for kp in self.gpt.iter_over_list(content)]
        print(','.join(kp_list), flush=True)
        return kp_list

    @torch.inference_mode()
    def detect_kps(self, tensor_images, kp):
        if self.molmo is None:
            self.molmo = self.Multimodal()

        all_kps = {}
        all_vis = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy())
            all_vis.append(image)
            kps = self.molmo.generated_kps_points(image, text=f"point to the {kp} on this image")
            try:
                kps, alt = self.molmo.parse_points_str(kps)
            except ParseError as e:
                pbar.clear()
                print(f'{kp}: Paring {str.strip(kps)} encountered {e}', file=sys.stderr)
                # Draw error text on image
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"No kps for {str.strip(kps)}\nError: {e}", fill='red')
                continue
            if len(kps) >= 8:
                pbar.clear()
                print(f'{kp}: too many kps for {alt}: {len(kps)}', file=sys.stderr)
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"Too many kps for {alt}", fill='red')
                continue
            if self.vis:
                self.molmo.draw_points(image, kps)
            all_kps[idx] = kps
        return all_kps, all_vis

    @torch.inference_mode()
    def backproject_kps(self, mesh, fragments, R, T, kps):
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
        depth = fragments.zbuf[..., 0] # N H W
        batch_size, imh, imw = depth.shape
        color = torch.zeros((batch_size, 3, imh, imw), dtype=torch.uint8, device=self.device)

        for idx, kp_item in kps.items():
            cur_color = Image.new(mode='RGB', size=(imw, imh))
            self.molmo.draw_points(cur_color, kp_item, radius=self.proj_radius, width=None, colors=self.COLOR_NAMES.values())
            cur_color = np.asarray(cur_color)
            cur_mask = np.any(cur_color, axis=-1)
            cur_idx = [self.COLOR_MAP.get(a.tobytes(), 0) for a in cur_color[cur_mask, :3]]
            alpha = torch.from_numpy(cur_color[cur_mask, -1])
            cur_mask = torch.from_numpy(cur_mask)
            color[idx, 0, cur_mask] = idx
            color[idx, 1, cur_mask] = torch.tensor(cur_idx, dtype=torch.uint8, device=self.device)
            color[idx, 2, cur_mask] = alpha.to(device=self.device, dtype=torch.uint8)

        mask = color[:, -1].bool()
        pts_3d = get_depth_point_cloud(cameras, depth.unsqueeze(1), mask.unsqueeze(1))   # N H W -> N 1 H W
        points, directions, origins = pts_3d.points_packed(), pts_3d.normals_packed(), pts_3d.features_packed()
        valid_mask = torch.bmm((points - origins).unsqueeze(-2), directions.unsqueeze(-1)).view(-1) > 0

        if self.vis:
            pts_3d_ref = get_rgbd_point_cloud(cameras, color, depth.unsqueeze(1), mask.unsqueeze(1))
            assert torch.allclose(pts_3d_ref.points_packed(), points[valid_mask])

        color = torch.cat([color.permute(0, 2, 3, 1)[mask],
                           valid_mask.to(dtype=color.dtype, device=color.device).unsqueeze(-1)], dim=-1)
        pts_3d = Pointclouds(
            points=points[np.newaxis],
            normals=directions[np.newaxis],
            features=color[np.newaxis])

        return pts_3d

    def filter_draw_invalid_kps_with_images(self, images_with_kps, pts_3d, kp, class_title, mesh_id):
        features = pts_3d.features_packed()
        valid_mask = features[..., -1].bool()
        valid_keypts = features[valid_mask, :2].view(dtype=torch.int16).unique().cpu().numpy()
        invalid_keypts = features[torch.logical_not(valid_mask), :2].view(dtype=torch.int16).unique().cpu().numpy()
        setdiff = np.setdiff1d(invalid_keypts, valid_keypts, assume_unique=True)

        for idx, count in Counter(x.tobytes()[0] for x in setdiff).items():
            ImageDraw.Draw(images_with_kps[idx]).text((10, 10), f"KPS back-projection failed {count} times", fill='red')

        print(f"Drawing {kp} in those images of {class_title}", flush=True)
        self.io.save_images(images_with_kps, class_title, mesh_id, postfix=kp, prefix='RawView')

        return valid_mask


    def collapse_same_color(self, pts_3d: Pointclouds):
        pts, color = pts_3d.points_packed(), pts_3d.features_packed()
        color_value = map(ImageColor.getrgb, self.color_names.values())
        color_value = torch.as_tensor(list(color_value), dtype=torch.float32, device=self.device)
        cluster = torch.isclose(color_value[:, None, :], color[..., :3], atol=1e-2).all(dim=-1)
        idx, = cluster.any(dim=-1).nonzero(as_tuple=True)
        mini_cluster = torch.stack([pts[cluster[i]].mean(dim=0) for i in idx])
        mini_color = torch.stack([color[cluster[i]].mean(dim=0) for i in idx])
        return Pointclouds(mini_cluster, features=mini_color)

    @torch.inference_mode()
    def aggregate_kps(self, mesh, pts_3d: Pointclouds, max_points=10000):
        features = pts_3d.features_packed()
        features_id, raw_weights, valid_mask = features[:, :2], features[:, 2], features[..., -1].bool()
        pts = pts_3d.points_packed()[valid_mask]

        if pts.size(0) <= 1:
            return Pointclouds(pts[None])

        per_point_mean_weight = raw_weights.mean(dtype=torch.float32)

        if pts.size(0) > max_points:
            pts = Pointclouds(pts.unsqueeze(0), features=features[valid_mask].unsqueeze(0)).subsample(max_points)
            features_id = pts.features_packed()[:, :2].view(torch.int16).view(-1)
            raw_weights = pts.features_packed()[:, 2]
            pts = pts.points_packed()
        else:
            raw_weights = raw_weights[valid_mask]
            features_id = features_id[valid_mask].view(torch.int16).view(-1)

        per_point_max_weight = max(raw_weights[features_id == m].sum() for m in features_id.unique())
        np_weights = (raw_weights / per_point_mean_weight).cpu().numpy()  # np_weights.max() ~= 4
        np_pts = pts.cpu().numpy()
        min_cluster_size = 10
        clustered_masks = None
        while not clustered_masks and min_cluster_size >= 2:
            # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1)
            clustering = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
            cluster_assign = clustering.fit_predict(np_pts, sample_weight=np_weights)
            clustered_masks = [cluster_assign == i for i in np.unique(cluster_assign) if i != -1]
            min_cluster_size = math.ceil(min_cluster_size / 2)

        clustered_rawpts = Pointclouds(points=[pts[mask] for mask in clustered_masks],
                                       features=[raw_weights[mask].unsqueeze(-1) for mask in clustered_masks])
        clustered_pts = Pointclouds([(p.points_packed() * p.features_packed()).sum(dim=0, keepdim=True) / total_w
                                     for p in clustered_rawpts if (total_w := p.features_packed().sum()) > 2 * per_point_max_weight])
        if not clustered_pts:
            clustered_pts = Pointclouds([(p.points_packed() * p.features_packed()).sum(dim=0, keepdim=True) / total_w
                                     for p in clustered_rawpts])
        return clustered_pts

    def init_kps_batch(self, tensor_images, kp, class_title, mesh_id):
        return {}

    def process_kp_list(self, mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=0):
        last_kp_cache = self.init_kps_batch(images, kp_list, class_title, mesh_id)

        prompt_semantic_list = defaultdict(list)
        for semantic_id, kp in kp_list.items():
            prompt_semantic_list[kp[prompt_idx]].append(semantic_id)

        all_kps = {}
        for kp_prompt, semantic_ids in prompt_semantic_list.items():
            if (kps_3d := last_kp_cache.get(kp_prompt, None)) is None:
                kps, images_with_kps = self.detect_kps(images, kp_prompt)
            else:
                kps = None if self.kp_initialized_empty else kps_3d

            if kps is not None:
                kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
                if self.vis:
                    valid_mask = self.filter_draw_invalid_kps_with_images(images_with_kps, kps_3d, kp_prompt, class_title, mesh_id)
                    kps_3d_raw = Pointclouds(
                        points=kps_3d.points_packed()[np.newaxis, valid_mask],
                        features=kps_3d.features_packed()[np.newaxis, valid_mask, :3].to(dtype=torch.uint8))
                    self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id,
                                     semantic_id=','.join(map(str, semantic_ids)),
                                     postfix=kp_prompt, prefix='RawPts')
                kps_3d = self.aggregate_kps(mesh, kps_3d)
                if self.vis:
                    self.io.save_kps(mesh, kps_3d, class_title, mesh_id,
                                     semantic_id=','.join(map(str, semantic_ids)),
                                     postfix=kp_prompt, prefix='Pts')
                if self.kp_initialized_empty:
                    last_kp_cache[kp_prompt] = kps_3d.cpu()
            else:
                assert kps_3d is not None

            all_kps[frozenset(semantic_ids)] = kps_3d

        return all_kps

    def main_loop(self, use_texture=False, save_rendered_images=debug_enabled(), batch_size=13):
        for mesh, keypoints, class_title, mesh_id, pcd in self.io.loop_over_test_datasets(use_texture):
            try:
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                # if mesh_id != 'e4e98f8654d29536dc858dada15498d2':
                #     continue
                if self.io.check_if_complete(kp_list, class_title, mesh_id):
                    print(f'Skipping class {class_title} mesh {mesh_id}')
                    continue

                print(f'Rendering class {class_title} mesh {mesh_id}')
                images, fragments, R, T = self.views_from_model(mesh, self.views, batch_size=batch_size, device=self.device)
                images = F.interpolate(images, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)

                # Convert to values between 0 and 255
                images = images * 255
                images.clamp_(0, 255)
                images = images.to(torch.uint8)
                if save_rendered_images:
                    self.io.save_images(images, class_title, mesh_id)

                # kp_list = self.get_kp_names(images)
                all_kps = self.process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id)
                self.io.save_kps_with_semantic_ids(mesh, all_kps, class_title, mesh_id)
            except IOError as e:
                print(f'Error with {e}', file=sys.stderr)


if __name__ == '__main__':
    KPNetGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ICCV25"),
                   expname='Rebuttal').main_loop(use_texture=False)
