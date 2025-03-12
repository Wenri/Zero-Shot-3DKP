import torch
import clip

import random

from einops import rearrange
from PIL import Image, ImageDraw
from matplotlib import colors as mcolors
from torch.nn import functional as F

from feature_backprojection.saliency_extractor import ViTExtractor


class RedCircle:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vit_extractor = ViTExtractor(device=self.device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @staticmethod
    def load_image(image_path):
        """Load an image from the given path."""
        image = Image.open(image_path).convert('RGB')
        return image

    @staticmethod
    def estimate_radius(image):
        """Estimate the radius of the circle."""
        return min(image.width, image.height) // 40

    def compute_clip_similarity(self, image, text_query):
        """Compute the CLIP similarity between an image and a text query."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text_query]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity

    def extract_saliency_maps(self, image, load_size=224, num_samples=100, mask=None):
        batch, _ = self.vit_extractor.preprocess(image, load_size=load_size)
        batch = self.vit_extractor.extract_saliency_maps(batch.to(self.device))

        h, w = self.vit_extractor.num_patches
        batch = rearrange(batch, '1 (h w) -> 1 1 h w', h=h, w=w)
        batch = F.interpolate(batch, size=(image.height, image.width), mode="bicubic", align_corners=False)
        if mask is not None:
            batch *= mask

        top_k = torch.topk(batch.view(-1), k=num_samples, largest=True)
        selection = torch.zeros_like(batch, dtype=batch.dtype).view(-1)
        selection[top_k.indices] = top_k.values

        selection = rearrange(selection, '(h w) -> h w', h=image.height, w=image.width)
        return selection.nonzero(), selection[selection.nonzero(as_tuple=True)]

    def randon_sample_mask(self, image, num_samples=100, mask=None):
        batch = torch.rand(image.height, image.width, device=mask.device)
        if mask is not None:
            batch *= mask

        top_k = torch.topk(batch.view(-1), k=num_samples, largest=True)
        selection = torch.zeros_like(batch, dtype=batch.dtype).view(-1)
        selection[top_k.indices] = top_k.values

        selection = rearrange(selection, '(h w) -> h w', h=image.height, w=image.width)
        return selection.nonzero()

    def find_best_circle_position(self, image, text_query, num_samples=100, radius=None, mask=None):
        """Find the circle position that maximizes the CLIP similarity with the text query."""
        width, height = image.size
        if radius is None:
            radius = self.estimate_radius(image)
        best_similarity = -1
        best_position = None
        if False:
            samples, _ = self.extract_saliency_maps(image, num_samples=num_samples, mask=mask)
        else:
            samples = self.randon_sample_mask(image, num_samples=num_samples, mask=mask)
        samples = samples.long().cpu()

        for y, x in samples.numpy().tolist():
            image_with_circle = self.draw_circle_on_image(image, x, y, radius=radius)
            similarity = self.compute_clip_similarity(image_with_circle, text_query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = (x / width, y / height)

        return best_position, best_similarity

    def draw_circle_on_image(self, image, x, y, radius=None, color='red', width=None, copy=True):
        """Draw a circle on the image at the specified coordinates."""
        if radius is None:
            radius = self.estimate_radius(image)
        image_with_circle = image.copy() if copy else image
        draw = ImageDraw.Draw(image_with_circle)
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=max(width or radius // 5, 5))
        return image_with_circle

    def draw_points(self, image, *kps, radius=5, width=2, colors=mcolors.TABLEAU_COLORS.values()):
        w, h = image.width, image.height
        for kp, color in zip(kps, colors):
            x1 = kp[0] * w
            y1 = kp[1] * h

            if radius:
                # Draw circles at the specified points (eyes)
                self.draw_circle_on_image(image, x1, y1, radius=radius, color=color, width=width, copy=False)
            else:
                # Create an ImageDraw object
                draw = ImageDraw.Draw(image)
                draw.point((x1, y1), fill=color)

    def main(self, image_path, text_query="a cat"):
        """Main function to execute the process."""

        # Load image
        image = self.load_image(image_path)

        # Find the best circle position
        best_position, best_similarity = self.find_best_circle_position(
            image, text_query, num_samples=100
        )

        print(f"Best position: {best_position}, similarity: {best_similarity}")

        # Draw the best circle and save the image
        self.draw_points(image, best_position)

        return image


def main():
    """Main function to execute the process."""
    # Replace with your image path
    image_path = "image.png"
    # Define text query
    text_query = "a cat"  # Replace with your text query

    red = RedCircle()
    image_with_best_circle = red.main(image_path, text_query=text_query)
    image_with_best_circle.save("image_with_best_circle.jpg")


if __name__ == "__main__":
    main()
