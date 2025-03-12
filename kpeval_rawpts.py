import json
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import IO
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from kp_utils import KeypointNetDataset, eval_iou, gen_geo_dists, CLASS_MAPPING


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
            for anno_dict in jdict.values():
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
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return False
        for semantic_id, kp in kp_list.items():
            if not mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists():
                return False
        return True

    def save_kps(self, mesh, kps_3d, class_title, mesh_id, semantic_id, postfix='keypts'):
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title, mesh_id, postfix='rawpts'):
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed().mean(dim=0, keepdim=True)
               for fname in save_dir.glob(f"{mesh_id}_*_{postfix}.ply")}
        return kps


class KPNetEvaluator:
    def __init__(self, log_dir: str | os.PathLike = Path(), expname='BackprojectPTS'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.io = KPNetIO(Path(log_dir, expname))
        self.ioref1 = KPNetIO(Path(log_dir, 'GPT-4o'))
        self.ioref2 = KPNetIO(Path(log_dir, 'GlobalPTS'))

    @staticmethod
    def find_nearest_idx(pcd, kps):
        kps, label = zip(*((v, torch.full(v.shape[:1], fill_value=int(k), dtype=torch.int)) for k, v in kps.items()))
        kps = torch.cat(kps, dim=0).cpu().numpy()
        label = torch.cat(label, dim=0).cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1).fit(pcd)
        indices = nbrs.kneighbors(kps, return_distance=False)
        return indices.squeeze(), label

    def main_loop(self, use_texture=False, offset=0, gaussian_sigma=0.01):
        for cat_name in ['airplane', 'chair', 'table']:  # CLASS_MAPPING.values()

            test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)
            geo_dists = {}

            pred_all_iou = {
                cat_name: {}
            }
            gt_all = {
                cat_name: {}
            }

            for i in tqdm(range(offset, len(test_dataset)), leave=False):
                mesh, keypoints, class_title, mesh_id, pcd = test_dataset[i]
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                if not self.io.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                if not self.ioref1.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                if not self.ioref2.check_if_complete(kp_list, class_title, mesh_id):
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
    KPNetEvaluator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                   expname='MolmoPTSNewPrompt').main_loop(
        gaussian_sigma=0.01)
