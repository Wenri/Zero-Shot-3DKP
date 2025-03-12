from pathlib import Path

import torch
from PIL import Image
from einops import rearrange

from clip_dinoiser.pointmodel import ClipDINOiser
from kpviews import KPNetGenerator


class RedCircleGenerator(KPNetGenerator):
    Multimodal = ClipDINOiser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = False

    def init_kps_batch(self, tensor_images, kp, cat):
        prompts = list(set(l[0] for l in kp.values()))
        models = self.molmo.load_model_against_prompts(prompts)
        images = [Image.fromarray(rearrange(img, 'c h w -> h w c').cpu().numpy()).convert('RGB') for img in
                  tensor_images]
        masks = tensor_images[:, -1].bool()
        # Find the best circle position
        best_position = self.molmo.visualize_per_image(images, masks, models)
        all_kps = {k: {i: best_position[i, j] for i in range(len(tensor_images)) if torch.any(best_position[i, j])} for
                   j, k in enumerate(prompts)}
        if self.vis:
            for k, kps in all_kps.items():
                for idx, mask in kps.items():
                    image = images[idx].copy()
                    self.molmo.draw_points(image, mask)
                    image.save(self.log_dir / f'{cat}_{k}_{idx}.png')
        return all_kps


if __name__ == '__main__':
    RedCircleGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                       expname='ClipDINOiser').main_loop(batch_size=7)
