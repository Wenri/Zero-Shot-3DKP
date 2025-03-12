from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw
from einops import rearrange
from tqdm import trange

from data_creation.pali_gemma import PaliGemma
from kp_utils.rendering import sample_view_points
from kpviews import KPNetGenerator

from pytorch3d.structures import Pointclouds

class PaliGemmaGenerator(KPNetGenerator):
    Multimodal = PaliGemma

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = True
        self.views = sample_view_points(self.dist, partition=1)

    def init_kps_batch(self, tensor_images, kp_list, class_title, mesh_id):
        prompts = {k for _, kp in kp_list.items() for k in kp}      
        all_kps = {v: Pointclouds(pts) for v in prompts if 
                   (pts := list(
            self.io.load_kps_with_semantic_ids(class_title, mesh_id, prefix='Pts', postfix=v).values()
            ))}
        return all_kps
    
    def detect_kps(self, tensor_images, kp):
        all_kps = {}
        all_vis = []
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            all_vis.append(image)
            try:  # Call the detect and segment instructions
                kps = self.molmo.generated_kps_points(
                    image, text=f"detect {kp}"
                ), self.molmo.generated_kps_points(
                    image, text=f"segment {kp}"
                )
            except Exception as e:
                print(f"Error segmenting {kp}: {e}")
                # Draw error text on image
                if self.vis:
                    ImageDraw.Draw(image).text((10, 10), f"No kps for {str.strip(kp)}\nError: {e}", fill='red')
                continue
            if self.vis:
                self.molmo.draw_points(image, kps)
            all_kps[idx] = kps
        return all_kps, all_vis


if __name__ == '__main__':
    PaliGemmaGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ICCV25"),
                       expname='PaliGemma').main_loop(batch_size=7)
