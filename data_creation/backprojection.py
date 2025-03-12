import sys
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights, look_at_view_transform, \
    CamerasBase, ray_bundle_to_ray_points, NDCMultinomialRaysampler
from pytorch3d.structures import Pointclouds

from sklearn.utils import gen_batches

from data_creation.keypointnet.utils.render_utils import Render
from kp_utils import setup_renderer


class RenderO3D(Render):
    def __init__(self, *args, res=512, **kwargs):
        """
        Setup a standard PyTorch3D renderer for rendering meshes.
        """
        self.o3d = setup_renderer(*args, res=res, **kwargs)
        super(RenderO3D, self).__init__(size=res, camera_poses=None)

    def __call__(self, meshes_world, cameras, **kwargs):
        # _, rendered_images, _, _, _ = self.render(
        #     path=None,
        #     clean=None,
        #     intensity=6.0,  # Light intensity
        #     mesh=mesh,
        #     only_render_images=True,
        #     color=None,
        #     correct_n=True,
        # )

        return self.o3d(meshes_world, cameras=cameras, **kwargs)


@torch.inference_mode()
def views_from_model(renderer, mesh, views, batch_size=None, device="cuda"):
    """
    Compute the features extracted by 'model' from rendered images rendered with 'renderer (deprecated)' for the keypoints

    :param renderer: pytorch3d MeshRendererWithFragments
    :param mesh: pytorch3d Mesh
    :param views: viewpoints (e.g. from 'views_around_object')
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :return: torch.Tensor point features (N, emb_dim)
    """
    with torch.no_grad():
        mesh = mesh.to(device=device)
        target = mesh.verts_packed().mean(dim=0, keepdim=True)
        views = torch.as_tensor(views, dtype=target.dtype, device=target.device) + target

    num_views = len(views)
    if batch_size is None:
        batch_size = num_views

    all_images = []
    all_fragments = defaultdict(list)

    with torch.no_grad():
        R, T = look_at_view_transform(eye=views, at=target, device=device)

        # 1. Render the mesh
        for s in gen_batches(num_views, batch_size):
            cameras = FoVPerspectiveCameras(R=R[s], T=T[s], device=device)
            lights = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=views[s], device=device)

            batch_images, fragments = renderer(mesh.extend(len(cameras)), cameras=cameras, lights=lights)
            all_images.append(batch_images)
            for k, v in vars(fragments).items():
                all_fragments[k].append(v)

        all_fragments = type(fragments)(**{k: torch.cat(v) for k, v in all_fragments.items()})
        return rearrange(torch.cat(all_images), 'b h w c -> b c h w'), all_fragments, R, T


def get_depth_point_cloud(
        camera: CamerasBase,
        depth_map: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        euclidean: bool = False,
):
    imh, imw = depth_map.shape[2:]

    # convert the depth maps to point clouds using the grid ray sampler
    ray_bundle = NDCMultinomialRaysampler(
            image_width=imw,
            image_height=imh,
            n_pts_per_ray=1,
            min_depth=1.0,
            max_depth=1.0,
            unit_directions=euclidean,
        )(camera)
    pts_3d = ray_bundle_to_ray_points(ray_bundle._replace(
        lengths=rearrange(depth_map, 'B 1 H W -> B H W 1')))

    mask = (depth_map > 0 if mask is None else mask).view(-1)

    return Pointclouds(points=pts_3d.view(-1, 3)[np.newaxis, mask],
                       normals=ray_bundle.directions.view(-1, 3)[np.newaxis, mask],
                       features=ray_bundle.origins.view(-1, 3)[np.newaxis, mask])


def debug_enabled():
    # Check environment variable
    if os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'):
        return True

    # Check debugger trace
    try:
        if sys.gettrace() is not None:
            return True
    except AttributeError:
        pass

    # Check sys.monitoring
    try:
        if sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None:
            return True
    except AttributeError:
        pass

    return False
