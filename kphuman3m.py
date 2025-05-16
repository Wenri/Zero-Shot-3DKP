import os
from pathlib import Path
import random
import numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
import torch
from data_creation.scene.cameras import colmap_to_pytorch3d, convert_camera_from_gs_to_pytorch3d
from kp_utils.data.utils import load_mesh
from kpeval import KPNetIO
from kpviews import KPNetGenerator
from data_creation.scene.dataset_readers import sceneLoadTypeCallbacks

class RealSceneIO(KPNetIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        colmap_data_path = os.environ.get("HUMAN3M_DATA_PATH")
        if colmap_data_path is None:
            raise ValueError("HUMAN3M_DATA_PATH is not set")
        self.colmap_data_path = Path(colmap_data_path)


    def loop_over_test_datasets(self, use_texture=False):
        scene_path = Path(self.colmap_data_path)
        mesh_id = scene_path.stem
        mesh = load_mesh(scene_path.absolute(), 'Scan', use_texture=False, use_normals=True, mesh_type='.obj')
        pcd = Pointclouds(*sample_points_from_meshes(mesh, return_normals=True))
        keypoints = ['eye', 'nose', 'mouth']
        yield mesh, keypoints, 'human3m', mesh_id, pcd

    def get_kp_names_from_lable(self, class_title, mesh_id, keypoints):
        return {idx: kp for idx, kp in enumerate(keypoints)}



class RealSceneGenerator(KPNetGenerator):
    KPIO = RealSceneIO

    def sample_view_points(self, dist, partition=None):
        return super().sample_view_points(dist, partition=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = True

    def process_kp_list(self, mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=slice(None)):
        return super().process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=prompt_idx)

if __name__ == '__main__':
    RealSceneGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ICCV25"),
                   expname='RebuttalReal').main_loop(use_texture=False)
