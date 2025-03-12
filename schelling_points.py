import json
import os
import sys
from collections import OrderedDict
from contextlib import suppress, nullcontext
from pathlib import Path
from types import NoneType
from xml.etree.ElementTree import ParseError

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor
from einops import rearrange
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from data_creation.backprojection import views_from_model, RenderO3D, debug_enabled
from data_creation.gpt4o import GPT4o
from data_creation.molmo import Molmo
from schelling_eval import SchellingIO
from kp_utils import sample_view_points


class Generator(RenderO3D):
    Multimodal = Molmo

    def __init__(self, log_dir: str | os.PathLike = Path(), expname='MolmoPTSNewPrompt'):
        self.log_dir = Path(log_dir)
        self.gpt = GPT4o()
        self.molmo = None
        self.io = SchellingIO(self.log_dir / expname)
        self.dist = 2.0
        self.res = 512
        self.scale = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.color_names = OrderedDict(a for a in mcolors.CSS4_COLORS.items() if any(ImageColor.getrgb(a[1])))
        self.vis = False
        super(Generator, self).__init__(self.device, res=self.scale * self.res)

    @torch.inference_mode()
    def get_prompts(self, tensor_images):
        image = Image.fromarray(rearrange(tensor_images[0], 'c h w -> h w c').cpu().numpy())
        response = self.gpt.get_kplist(image)
        content = json.loads(response.choices[0].message.content)
        kp_list = [kp for kp in self.gpt.iter_over_list(content)]
        print(','.join(kp_list), flush=True)
        return kp_list

    @torch.inference_mode()
    def detect_kps(self, tensor_images, kp, cat):
        if self.molmo is None:
            self.molmo = self.Multimodal()

        all_kps = {}
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy())
            kps = self.molmo.generated_kps_points(image, text=f"point to the {kp} on this {cat}")
            try:
                kps, alt = self.molmo.parse_points_str(kps)
            except ParseError as e:
                pbar.clear()
                print(f'{kp}: Paring {str.strip(kps)} encountered {e}', file=sys.stderr)
                continue
            if len(kps) >= 8:
                pbar.clear()
                print(f'{kp}: too many kps for {alt}: {len(kps)}', file=sys.stderr)
                continue
            if self.vis:
                print(f"Drawing {alt} in this image", flush=True)
                self.molmo.draw_points(image, kps)
                image.show()
            all_kps[idx] = kps
        return all_kps

    @torch.inference_mode()
    def backproject_kps(self, mesh, fragments, R, T, kps):
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
        depth = rearrange(fragments.zbuf[..., 0], 'N H W -> N 1 H W')
        mask = torch.zeros_like(depth, dtype=torch.bool)
        color = torch.zeros((depth.size(0), 4, depth.size(2), depth.size(3)), dtype=torch.float32, device=self.device)

        imh, imw = mask.shape[2:]
        color_iter = iter(self.color_names.values())
        for idx, kp_item in kps.items():
            cur_color = Image.new(mode='RGB', size=(imw, imh))
            self.molmo.draw_points(cur_color, kp_item, radius=None, colors=color_iter)
            cur_color = rearrange(np.asarray(cur_color), 'H W C -> C H W')
            mask[idx, 0, np.any(cur_color > 0, axis=0)] = 1
            color[idx, :3] = torch.from_numpy(cur_color)

        color[:, 3] = depth[:, 0]
        pts_3d = get_rgbd_point_cloud(cameras, color, depth, mask)

        if self.vis:
            self.io.save_mesh(mesh, self.log_dir / "output_mesh.ply")
            self.io.save_pointcloud(Pointclouds(
                pts_3d.points_packed()[None]), self.log_dir / "output_pointcloud.ply")
        return pts_3d

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
    def aggregate_kps(self, mesh, pts_3d: Pointclouds):
        pts = pts_3d.points_packed()

        # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1)
        clustering = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.1, allow_single_cluster=True)
        cluster_assign = clustering.fit_predict(pts.cpu().numpy())
        clustered_pts = Pointclouds(torch.stack(
            [pts[cluster_assign == i].mean(dim=0) for i in np.unique(cluster_assign)])[None])

        if self.vis:
            self.io.save_pointcloud(clustered_pts, self.log_dir / "output_pointcloud_c.ply")
        return clustered_pts

    def main_loop(self,vis=False, batch_size=1, debug=debug_enabled()):
        views = sample_view_points(self.dist, 3)
        for mesh, class_title, mesh_id, _ in self.io.loop_over_test_datasets():
            with nullcontext() if True else suppress(Exception):
                
                print(f'Rendering class {class_title} mesh {mesh_id}')
                images, fragments, R, T = views_from_model(self, mesh, views, batch_size=batch_size, device=self.device)
                images = F.interpolate(images, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
                if vis:
                    plt.imshow(rearrange(images[0], 'c h w -> h w c').clamp(min=0, max=1).cpu().numpy())
                    plt.show()
                # Convert to values between 0 and 255
                images = images * 255
                images.clamp_(0, 255)
                images = images.to(torch.uint8)
                kp_list = self.get_prompts(images)

                last_kp = {}
                for kp in kp_list:
                    if (kps_3d := last_kp.get(kp[-1], None)) is None:
                        kps = self.detect_kps(images, kp[-1], class_title)
                        kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
                        kps_3d_raw = Pointclouds(points=kps_3d.points_packed()[None],
                                                 features=kps_3d.features_packed()[None, ..., :3].to(dtype=torch.uint8))
                        self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id, "schelling", postfix='rawpts')
                        kps_3d = self.aggregate_kps(mesh, kps_3d)
                        last_kp[kp[0]] = kps_3d.cpu()
                    self.io.save_kps(mesh, kps_3d, class_title, mesh_id, "schelling")


if __name__ == '__main__':
    Generator(Path.home().joinpath("Desktop", "Zero-Shot-3DKP", "SchellingPointsResults")).main_loop()
