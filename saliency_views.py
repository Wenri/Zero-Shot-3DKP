from pathlib import Path

import torch
import torchvision
from PIL import Image
from einops import rearrange
from matplotlib import colors as mcolors
from tqdm import trange

from data_creation.red_circle import RedCircle
from kpviews import KPNetGenerator


class DINOSaliency(RedCircle):
    def find_salient_position(self, image, num_samples=100, mask=None):
        """Find the circle position that maximizes the CLIP similarity with the text query."""
        width, height = image.size
        best_position = []
        bbox_w = 0.01 * min(width, height)
        samples, scores = self.extract_saliency_maps(image, num_samples=num_samples, mask=mask)
        bboxes = torch.concat((samples - bbox_w, samples + bbox_w), dim=-1)
        bboxes = torchvision.ops.nms(boxes=bboxes, scores=scores, iou_threshold=0.5)
        samples = samples.long()

        for y, x in samples[bboxes].cpu().numpy().tolist():
            best_position.append((x / width, y / height))

        return best_position

    def draw_points(self, image, kps, radius=5, width=2, colors=mcolors.TABLEAU_COLORS.values()):
        return super().draw_points(image, *kps, radius=radius, width=width, colors=colors)


class SaliencyGenerator(KPNetGenerator):
    Multimodal = DINOSaliency

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = False

    @torch.inference_mode()
    def get_kp_names(self, tensor_images):
        return ['salient points']

    def detect_kps(self, tensor_images, kp, cat):
        all_kps = {}
        pbar = trange(tensor_images.size(0))
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            mask = tensor_images[idx, -1].bool()
            # Find the best circle position
            best_position = self.molmo.find_salient_position(image, num_samples=100, mask=mask)
            all_kps[idx] = best_position
            if self.vis:
                print(f"Drawing salient position in this image", flush=True)
                self.molmo.draw_points(image, best_position)
                image.show()
        return all_kps


if __name__ == '__main__':
    SaliencyGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                      expname='Saliency').main_loop(batch_size=7)
