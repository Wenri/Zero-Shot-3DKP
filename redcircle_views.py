from pathlib import Path

from PIL import Image
from einops import rearrange
from tqdm import trange

from data_creation.red_circle import RedCircle
from kpviews import KPNetGenerator


class RedCircleGenerator(KPNetGenerator):
    Multimodal = RedCircle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = False

    def detect_kps(self, tensor_images, kp, cat):
        all_kps = {}
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            mask = tensor_images[idx, -1].bool()
            # Find the best circle position
            best_position, best_similarity = self.molmo.find_best_circle_position(
                image, kp, num_samples=100, mask=mask
            )
            all_kps[idx] = best_position
            if self.vis:
                print(f"Drawing {kp} in this image", flush=True)
                self.molmo.draw_points(image, best_position)
                image.show()
        return all_kps


if __name__ == '__main__':
    RedCircleGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                       expname='RedCircleGrid').main_loop(batch_size=7)
