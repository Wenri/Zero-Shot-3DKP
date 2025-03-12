from pathlib import Path

import torch
from PIL import Image
from einops import rearrange
from pytorch3d.structures import Pointclouds
from tqdm import trange

from data_creation.red_circle import RedCircle
from kpviews import KPNetGenerator
from unsupervised_keypoints import MainKeypointRegressor


class StableKeypoints(KPNetGenerator):
    Multimodal = MainKeypointRegressor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = True

    def init_kps_batch(self, tensor_images, kp, cat):
        images = [Image.fromarray(rearrange(img, 'c h w -> h w c').cpu().numpy()) for img in
                  tensor_images]
        num_points = len(kp)
        points = self.molmo.get_embeddings(images, top_k=num_points)
        all_kps = {k: {i: points[i, j] for i in range(len(tensor_images))} for j, k in enumerate(kp)}
        if self.vis:
            for k, kps in all_kps.items():
                for idx, pts in kps.items():
                    image = images[idx].copy()
                    self.molmo.draw_points(image, pts)
                    image.save(self.log_dir / f'{cat}_{k}_{idx}.png')
        return all_kps

    def process_kp_list(self, mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=0):
        last_kp = self.init_kps_batch(images, kp_list, class_title)
        for semantic_id, kps in last_kp.items():
            kps_3d = self.backproject_kps(mesh, fragments, R, T, kps)
            kps_3d_raw = Pointclouds(points=kps_3d.points_packed()[None],
                                     features=kps_3d.features_packed()[None, ..., :3].to(dtype=torch.uint8))
            self.io.save_kps(mesh, kps_3d_raw, class_title, mesh_id, semantic_id, postfix='rawpts')
            kps_3d = self.aggregate_kps(mesh, kps_3d)

            self.io.save_kps(mesh, kps_3d, class_title, mesh_id, semantic_id)

            self.vis = False


if __name__ == '__main__':
    StableKeypoints(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                    expname='StableKeypoints').main_loop(batch_size=7)
