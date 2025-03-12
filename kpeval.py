import itertools
import json
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from einops import rearrange
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds, join_pointclouds_as_scene
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from kp_utils import KeypointNetDataset, eval_iou, gen_geo_dists, CLASS_MAPPING


def setids_to_uint8_flags(setids):
    flags = 0
    for id in setids:
        if not isinstance(id, int) or id >= 24 or id < 0:
            raise ValueError(f"ID {id} must be an integer between 0 and 23")
        flags |= (1 << id)  # Set the bit at position id
    return torch.frombuffer(flags.to_bytes(length=3, byteorder='big'), dtype=torch.uint8)

class KPNetIO(IO):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.jdict = self.load_semantic_names()
        super(KPNetIO, self).__init__()

    @staticmethod
    def load_semantic_names():
        jdict = defaultdict(dict)
        with Path('data_creation', 'keypointnet', 'semantic_names.json').open('r') as f:
            j = json.load(f)
        for item in j:
            jdict[CLASS_MAPPING[item['class_id']]][item['model_id']] = item['semantic_id_mapping']
        return jdict

    def get_kp_names_from_lable(self, class_title, mesh_id, keypoints):
        jdict = self.jdict.get(class_title)
        ret_dict = defaultdict(list)

        for kps in keypoints:
            semantic_id = kps['semantic_id']
            ret_dict[semantic_id].append(kps['label'][0])
            for anno_dict in jdict.values() if jdict is not None else ():
                if anno_str := anno_dict.get(str(semantic_id), None):
                    if anno_str == 'None':
                        continue
                    ret_dict[semantic_id].append(anno_str)
        return ret_dict

    @staticmethod
    def loop_over_test_datasets(use_texture):
        test_datasets = (
            KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture) for cat_name in
            ['airplane', 'chair', 'table']  # CLASS_MAPPING.values()
        )
        for batch in zip_longest(*test_datasets):
            yield from filter(None, batch)

    def check_if_complete(self, kp_list, class_title, mesh_id):
        save_dir = self.output_dir / class_title / mesh_id
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return self.check_if_complete2(kp_list, class_title, mesh_id)
        all_semantic_ids = setids_to_uint8_flags(kp_list.keys())
        semantic_id='0x{:02X}{:02X}{:02X}'.format(*all_semantic_ids.tolist())
        return mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists()
    
    def check_if_complete2(self, kp_list, class_title, mesh_id):
        save_dir = self.output_dir / class_title
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return False
        for semantic_id, kp in kp_list.items():
            if not mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists():
                return False
        return True

    def save_kps(self, mesh, kps_3d, class_title, mesh_id, semantic_id, postfix='keypts', prefix=None):
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        kps_3d = join_pointclouds_as_scene(kps_3d)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{prefix or mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title, mesh_id, postfix='keypts', prefix=None):
        save_dir = self.output_dir / class_title / mesh_id
        if not save_dir.exists():
            return self.load_kps2(class_title, mesh_id, postfix)
        all_kps = {int.from_bytes(bytes.fromhex(fname.stem.split('_')[1].removeprefix('0x')), 'big'):
                   self.load_pointcloud(fname).points_packed()
                   for fname in save_dir.glob(f"{prefix or mesh_id}_0x*_{postfix}.ply")}
        return all_kps
    
    def load_kps2(self, class_title, mesh_id, postfix='keypts'):
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed()
               for fname in save_dir.glob(f"{mesh_id}_*_{postfix}.ply")}
        return kps
    
    def load_kps_with_semantic_ids(self, class_title, mesh_id, postfix='keypts',prefix=None):
        save_dir = self.output_dir / class_title / mesh_id
        all_kps = {frozenset(map(int, fname.stem.split('_')[1].split(','))): 
                   self.load_pointcloud(fname).points_packed()
                   for fname in save_dir.glob(f"{prefix or mesh_id}_*_{postfix}.ply")}
        return all_kps
    
    def save_kps_with_semantic_ids(self, mesh, all_kps, class_title, mesh_id, postfix='keypts', prefix=None):
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)

        all_kps_3d = []
        all_flags = []
        for semantic_ids, kps in all_kps.items():
            pts = kps.points_packed()
            all_kps_3d.append(pts.cpu())
            # Create feature tensor same shape as points
            flags = setids_to_uint8_flags(semantic_ids)
            all_flags.append(flags.expand(pts.shape[0], -1))

        all_semantic_ids = setids_to_uint8_flags(itertools.chain.from_iterable(all_kps.keys()))
        kps_3d = Pointclouds(points=all_kps_3d, features=all_flags)
        self.save_kps(mesh, kps_3d, class_title, mesh_id,
                      semantic_id='0x{:02X}{:02X}{:02X}'.format(*all_semantic_ids.tolist()),
                      postfix=postfix, prefix=prefix)

    def save_images(self, images, class_title, mesh_id, postfix='views', prefix=None):
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(images, torch.Tensor):
            images = rearrange(images, 'b c h w -> b h w c').cpu()
            images = [Image.fromarray(img.numpy()) for img in images]

        num_images = len(images)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))

        plt.ioff()
        plt.figure(figsize=(4*cols, 4*rows))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix or mesh_id}_{postfix}.png", bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()


class KPNetEvaluator:
    def __init__(self, log_dir: str | os.PathLike = Path(), expname='BackprojectPTS', ioref=()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.io = KPNetIO(Path(log_dir, expname))
        self.ioref = [KPNetIO(Path(log_dir, refname)) for refname in ioref]

    @staticmethod
    def find_nearest_idx(pcd, kps):
        kps, label = zip(*((v, torch.full(v.shape[:1], fill_value=int(k), dtype=torch.int)) for k, v in kps.items()))
        kps = torch.cat(kps, dim=0).cpu().numpy()
        label = torch.cat(label, dim=0).cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1).fit(pcd)
        indices = nbrs.kneighbors(kps, return_distance=False)
        return indices.squeeze(), label

    def main_loop(self, use_texture=False, offset=0, gaussian_sigma=0.01):
        for cat_name in ['table', 'airplane', 'chair']:  # CLASS_MAPPING.values()

            test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)
            geo_dists = {}

            pred_all_iou = {
                cat_name: {}
            }
            gt_all = {
                cat_name: {}
            }

            pbar = tqdm(range(offset, len(test_dataset)), leave=False)
            for i in pbar:
                mesh, keypoints, class_title, mesh_id, pcd = test_dataset[i]
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                if not self.io.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                for ioref in self.ioref:
                    if not ioref.check_if_complete(kp_list, class_title, mesh_id):
                        continue

                if mesh_id not in pred_all_iou[cat_name]:
                    pred_all_iou[cat_name][mesh_id] = {}
                    pred_all_iou[cat_name][mesh_id]["indices"] = []
                    pred_all_iou[cat_name][mesh_id]["confidence"] = []
                if mesh_id not in gt_all[cat_name]:
                    gt_all[cat_name][mesh_id] = []
                pcd = pcd.points_packed().numpy().astype(np.float32)

                geo_dists[mesh_id] = gen_geo_dists(pcd).astype(np.float32)

                geo_dists_mesh = geo_dists[mesh_id]
                geo_dists_mesh[np.isinf(geo_dists_mesh)] = geo_dists_mesh[~np.isinf(geo_dists_mesh)].max()
                normalized_geo_dists = geo_dists_mesh / np.max(geo_dists_mesh)

                kps = self.io.load_kps(class_title, mesh_id)
                predictions_idx, _ = self.find_nearest_idx(pcd, kps)
                # predictions_idx, selection_matrix = optimize_keypoint_candidates(kp_dists, normalized_geo_dists,
                #                                                                  kp_features, features,
                #                                                                  num_steps=5000, lr=0.1, device=device,
                #                                                                  dist_alpha=dist_weight,
                #                                                                  selection_beta=0)

                # pred_all_iou[cat_name][mesh_id]["confidence"].extend(selection_matrix[:, :-1].max(axis=0).tolist())
                pred_all_iou[cat_name][mesh_id]["indices"].extend(predictions_idx.tolist())

                for kp in keypoints:
                    gt_all[cat_name][mesh_id].append(kp["pcd_info"]["point_index"])

                pbar.set_description(f"Processing {cat_name} mIoU-0.1: %({cat_name})s" % eval_iou(
                    pred_all_iou, gt_all, geo_dists, dist_thresh=0.1))

            print(cat_name)
            for i in range(21):
                dist_thresh = 0.01 * i
                iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
                iou_l = list(iou.values())
                s = ""
                for x in iou_l:
                    s += "{}\t".format(x)
                print('mIoU-{}: {}'.format(dist_thresh, s))


if __name__ == '__main__':
    KPNetEvaluator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ICCV25"),
        expname='PaliGemma',
        ioref=()
    ).main_loop(gaussian_sigma=0.01)
