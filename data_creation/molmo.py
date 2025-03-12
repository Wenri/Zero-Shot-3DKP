import math
import time
from collections import defaultdict, OrderedDict

import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image, ImageDraw
import requests
import torch
from xml.etree import ElementTree
from matplotlib import colors as mcolors
from einops import reduce

class Molmo:
    COLOR_NAMES = ('tab:red', 'tab:orange', 'tab:purple', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
    COLOR_VALUES = tuple(map(mcolors.TABLEAU_COLORS.get, COLOR_NAMES))

    def __init__(self, model_path='allenai/Molmo-7B-D-0924'):
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,  # 'allenai/Molmo-72B-0924'
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

    @torch.inference_mode()
    def image_text_token(self, image, text='point to the salient keypoints in this image'):
        # process the image and text
        inputs = self.processor.process(
            images=[image],
            text=text
        )

        return inputs

    @torch.inference_mode()
    def generated_kps_points(self, image, text):
        inputs = self.image_text_token(image, text=text)

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        return generated_text
        # >>> This image features an adorable black Labrador puppy sitting on a wooden deck.
        #     The puppy is positioned in the center of the frame, looking up at the camera...

    @staticmethod
    def parse_points_str(input_str):
        # Parse XML
        root = ElementTree.fromstring(input_str)
        alt = root.attrib.get('alt', None)
        kps = defaultdict(dict)
        for k, v in root.attrib.items():
            if k.startswith(("x", "y")):
                kps[k[1:]][k[0]] = float(v)
        return kps, alt

    @staticmethod
    def draw_points(image, kps, kps_wh=(100, 100), radius=5, width=2, colors=COLOR_VALUES):
        w, h = image.width, image.height
        k_w, k_h = kps_wh
        # Create an ImageDraw object
        draw = ImageDraw.Draw(image)
        L = torch.zeros((h, w))
        for kp, color in zip(kps.values(), colors):
            x1 = float(kp['x']) / k_w * w
            y1 = float(kp['y']) / k_h * h

            if radius:
                # Draw circles at the specified points (eyes)
                if width:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=color, width=width)
                else:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color)
                    alpha = Image.new(mode='L', size=(w, h))
                    ImageDraw.Draw(alpha).ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=1)
                    alpha = torch.from_numpy(np.asarray(alpha))
                    pos_y, pos_x = alpha.nonzero(as_tuple=True)
                    sigma = radius / 3
                    pdf = torch.sqrt((pos_x - x1) ** 2 + (pos_y - y1) ** 2)
                    pdf = torch.exp(-0.5 * (pdf / sigma)**2) / (sigma * math.sqrt(2*torch.pi))
                    L[pos_y, pos_x] += pdf / torch.max(pdf)
            else:
                draw.point((x1, y1), fill=color)

        if radius and not width:
            L = Image.fromarray(torch.clamp(L * 255, 0, 255).to(dtype=torch.uint8).numpy())
            image.putalpha(L)
        return image


def main():
    molmo = Molmo()
    image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
    t0 = time.time()
    generated_text = molmo.generated_kps_points(image,
                                                text='point to the Paws in this image, where Small, visible on the wooden floor')
    t1 = time.time()
    print(t1 - t0)
    molmo.draw_points(image, generated_text)


if __name__ == "__main__":
    main()
