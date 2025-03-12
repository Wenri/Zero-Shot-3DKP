import base64
import io
import json
import sys
from pathlib import Path

import requests
from einops import rearrange
from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import trange

from data_creation.molmo import Molmo
from kpviews import KPNetGenerator

from matplotlib import colors as mcolors


def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    # Save the image to the buffer in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_data = buffered.getvalue()
    # Encode the byte data in base64
    base64_encoded = base64.b64encode(img_data).decode('utf-8')
    return base64_encoded


class GPT4oLocalize:
    def __init__(self):
        self.image_wh = None

    def get_kplist(self, client, image, kp):
        self.image_wh = image.width, image.height
        image_encoded = pil_to_base64(image)
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_encoded}"}
                    },
                    {
                        "type": "text",
                        "text": f"Point to the {kp} in this image. Output the point locations in the image (xy) coordinate system"
                    }
                ]
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "json_object"
            }
        )
        return response

    def norm_location_list(self, content, name=None):
        for partname, location in content.items():
            if isinstance(location, list):
                yield from enumerate(location)
            elif isinstance(location, dict):
                yield from self.norm_location_list(location, partname)
            else:
                yield name, content
                break

    def draw_points(self, image, kps, kps_wh=None, radius=5, width=2, colors=mcolors.TABLEAU_COLORS.values()):
        w, h = image.width, image.height
        k_w, k_h = kps_wh if kps_wh is not None else self.image_wh

        for kp, color in zip(kps.values(), colors):
            try:
                x1 = kp['x']
                y1 = kp['y']
            except Exception:
                try:
                    x1, y1 = kp.values()
                except Exception:
                    x1, y1 = kps.values()

            x1 = float(x1) / k_w * w
            y1 = float(y1) / k_h * h

            # Create an ImageDraw object
            draw = ImageDraw.Draw(image)

            if radius:
                # Draw circles at the specified points (eyes)
                if width:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=color, width=width)
                else:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color)
            else:
                draw.point((x1, y1), fill=color)


class GPT4oGenerator(KPNetGenerator):
    Multimodal = GPT4oLocalize

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis = False

    def detect_kps(self, tensor_images, kp, cat):
        all_kps = {}
        pbar = trange(tensor_images.size(0), desc=kp)
        for idx in pbar:
            image = Image.fromarray(rearrange(tensor_images[idx], 'c h w -> h w c').cpu().numpy()).convert('RGB')
            # Find the best circle position
            response = self.molmo.get_kplist(self.gpt.client, image, kp)
            try:
                content = json.loads(response.choices[0].message.content)
                kps = {k: v for k, v in self.molmo.norm_location_list(content)}
            except Exception as e:
                pbar.clear()
                print(f'{kp}: Paring {response} encountered {e}', file=sys.stderr)
                continue
            all_kps[idx] = kps
            if self.vis:
                print(f"Drawing {kp} in this image", flush=True)
                self.molmo.draw_points(image, kps)
                image.show()
        return all_kps


if __name__ == '__main__':
    GPT4oGenerator(Path.home().joinpath("pCloudDrive", "ResearchProjects", "ProjectingFeatures"),
                   expname='GPT-4o').main_loop(batch_size=7)
