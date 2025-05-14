import os
from pathlib import Path
import numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
import torch
from data_creation.scene.cameras import colmap_to_pytorch3d, convert_camera_from_gs_to_pytorch3d
from kp_utils.data.utils import load_mesh
from kpeval import KPNetIO
from kpviews import KPNetGenerator
from data_creation.scene.dataset_readers import sceneLoadTypeCallbacks

class RealSceneIO(KPNetIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        colmap_data_path = os.environ.get("COLMAP_DATA_PATH")
        if colmap_data_path is None:
            raise ValueError("COLMAP_DATA_PATH is not set")
        colmap_data_path = Path(colmap_data_path)
        self.scene_info = sceneLoadTypeCallbacks["Colmap"](colmap_data_path, images=None, eval=False)


    def loop_over_test_datasets(self, use_texture=False):
        scene_path = Path(self.scene_info.ply_path).parents[2]
        mesh_id = scene_path.stem
        mesh = load_mesh(scene_path.absolute(), 'refined_mesh', use_texture=True)
        pcd = self.scene_info.point_cloud
        pcd = Pointclouds(points=torch.from_numpy(pcd.points[np.newaxis]),
                          normals=torch.from_numpy(pcd.normals[np.newaxis]),
                          features=torch.from_numpy(pcd.colors[np.newaxis]))
        keypoints = ['testing']
        yield mesh, keypoints, 'colmap', mesh_id, pcd

    def get_kp_names_from_lable(self, class_title, mesh_id, keypoints):
        return keypoints



class RealSceneGenerator(KPNetGenerator):
    KPIO = RealSceneIO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, res=(540, 960), **kwargs)
        scene_info = self.io.scene_info
        self.views = scene_info.train_cameras + scene_info.test_cameras
        self.views = self.views[:1]
        self.vis = True

    def views_from_model(self, mesh, views, batch_size=None, device="cuda"):
        # R, T = zip(*[colmap_to_pytorch3d(
        #     rotation=torch.from_numpy(view.R).to(device=device),
        #     translation=torch.from_numpy(view.T).unsqueeze_(-1).to(device=device),
        #     device=device) for view in self.views])
        # FoV = [view.FovY for view in self.views]

        # cameras = FoVPerspectiveCameras(R=torch.stack(R), T=torch.stack(T), fov=FoV, degrees=False, device=device)
        cameras = convert_camera_from_gs_to_pytorch3d(self.views)
        ret = super().views_from_model(mesh, cameras, batch_size=batch_size, device=device)
        return ret


if __name__ == '__main__':
    RealSceneGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ICCV25"),
                   expname='RebuttalReal').main_loop(use_texture=False)
