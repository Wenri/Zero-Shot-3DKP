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

from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, FoVPerspectiveCameras, look_at_view_transform, PointLights, TexturesVertex
from pytorch3d.utils import ico_sphere


class RenderO3D(Render):
    setup_renderer = staticmethod(setup_renderer)

    def __init__(self, *args, res=512, **kwargs):
        """
        Setup a standard PyTorch3D renderer for rendering meshes.
        """
        self.o3d = self.setup_renderer(*args, res=res, **kwargs)
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
def camera_from_eye_at_up(eye, at, device="cuda"):
    R, T = look_at_view_transform(eye=eye, at=at, device=device)
    rotate = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]).float().to(R.device)
    R = rotate @ R
    return FoVPerspectiveCameras(R=R, T=T, device=device)


@torch.inference_mode()
def views_from_model(renderer, mesh, views, sphere_location=None,sphere_radius=0.05,batch_size=None, device="cuda"):
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

    mesh = mesh.to(device=device)
    if not isinstance(views, CamerasBase):
        with torch.no_grad():
            target = mesh.verts_packed().mean(dim=0, keepdim=True)
            views = torch.as_tensor(views, dtype=target.dtype, device=target.device) + target
        views = camera_from_eye_at_up(views, target, device=device)

    num_views = len(views)
    if batch_size is None:
        batch_size = num_views

    if sphere_location is not None:
        # Create a red sphere at the specified location
        sphere_mesh = ico_sphere(4, device=device)  # Using a high-resolution icosphere
        sphere_mesh = sphere_mesh.scale_verts(sphere_radius)  # Scale to the desired radius
        sphere_mesh = sphere_mesh.offset_verts(sphere_location.to(device))  # Move it to the desired location

        # Create red color for the sphere vertices
        red_color = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(sphere_mesh.verts_packed().shape[0], -1)
        sphere_mesh.textures = TexturesVertex(verts_features=red_color[None])

        # Combine the main mesh and the sphere mesh
        mesh = Meshes(
            verts=[torch.cat([mesh.verts_packed(), sphere_mesh.verts_packed()])],
            faces=[torch.cat([mesh.faces_packed(), sphere_mesh.faces_packed() + mesh.verts_packed().shape[0]])],
            textures=TexturesVertex(verts_features=torch.cat([mesh.textures.verts_features_packed(), red_color])[None])
        )

    all_images = []
    all_fragments = defaultdict(list)

    with torch.no_grad():
        # 1. Render the mesh
        for s in gen_batches(num_views, batch_size):
            cameras = views[torch.as_tensor(range(s.start, s.stop))]
            lights = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=cameras.get_camera_center(), device=device)

            batch_images, fragments = renderer(mesh.extend(len(cameras)), cameras=cameras, lights=lights)
            all_images.append(batch_images)
            for k, v in vars(fragments).items():
                all_fragments[k].append(v)

        all_fragments = type(fragments)(**{k: torch.cat(v) for k, v in all_fragments.items()})
        return rearrange(torch.cat(all_images), 'b h w c -> b c h w'), all_fragments, cameras.R, cameras.T


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
