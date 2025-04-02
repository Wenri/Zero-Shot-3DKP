import math
import time
from collections import defaultdict, OrderedDict

import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image, ImageDraw
import requests
import torch
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from matplotlib import colors as mcolors
from einops import reduce, rearrange

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
        self.block_handles = OrderedDict()
        self.emb_handles = OrderedDict()
        self.register_hooks()

    def register_hooks(self):
        for block in self.model.model.transformer.blocks:
            self.emb_handles[block.rotary_emb] = self.block_handles[block] = {
                'handle_emb': block.rotary_emb.register_forward_hook(self.attention_emb_hook, with_kwargs=True),
                'handle_block': block.register_forward_hook(self.attention_block_hook, with_kwargs=True),
                'attn_weight': []
            }

    def attention_emb_hook(self, module, args, kwargs, output):
        q, k = output
        record = self.emb_handles[module]
        record['q'] = q
        record['k'] = k

    def attention_block_hook(self, module, args, kwargs, output):
        record = self.block_handles[module]
        if (layer_past := kwargs['layer_past']) is not None:
            q, k = record.pop('q'), record.pop('k')
            past_key, _ = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attention_bias = module._cast_attn_bias(
                kwargs['attention_bias'][:, :, key_len - query_len : key_len, :key_len], k.dtype
            )
            record['attn_weight'].append(self.calc_attention(q, k, attention_bias))

    def calc_attention(self, q, k, attn_bias):
        num_q_heads = q.size(1)
        num_k_heads = k.size(1)

        if num_q_heads != num_k_heads:
            assert num_q_heads % num_k_heads == 0
            k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1, output_size=num_q_heads)

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight

    def collect_attn_weights(self, output):
        attention_weights = [record.pop('attn_weight') for record in self.block_handles.values()]
        last_attn_weight = attention_weights[-1]
        return last_attn_weight

    def rearrange_features_to_patches(self, images):
        images = rearrange(images, f'1 b (ph pw) ... -> b ph pw ...', ph=24, pw=24)
        return images

    def rearrange_patches_to_images(self, images, n_channels=3):
        h, w, c = 'h w c'.split() if n_channels else ('', '', '')
        extra_dims = {h:14, w:14, c:n_channels} if n_channels else {}
        images = rearrange(images, f'b (hl dh) (wl dw) ({h} {w} {c}) ...  -> b (hl wl) (dh {h}) (dw {w}) {c} ...',
                           dh=2, dw=2, **extra_dims)
        return images

    def visualize_patched_images(self, images, mask):
        images = self.rearrange_patches_to_images(images)
        mask = self.rearrange_patches_to_images(mask.unsqueeze(-1), n_channels=None).any(dim=(-2,-1))
        images = images * torch.as_tensor(OPENAI_CLIP_STD, device=images.device)
        images += torch.as_tensor(OPENAI_CLIP_MEAN, device=images.device)
        images = images[mask]

        import matplotlib.pyplot as plt
        # Create figure and axes for 48x12 grid
        fig, ax = plt.subplots(figsize=(6, 24))  # Adjusted figure size for vertical layout

        # Plot each image patch
        for i in range(len(images)):
            img = images[i].cpu().numpy()
            h, w = i // 12, i % 12  # Changed to 12 columns
            ax.imshow(img, extent=[w, w+1, 47-h, 48-h])  # Adjusted height to 48

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 12)  # Changed width to 12
        ax.set_ylim(0, 48)  # Changed height to 48
        plt.show(block=True)
        return images, mask

    def plot_attn_weights(self, inputs, generated_text, attn_weights):
        images = self.rearrange_features_to_patches(inputs['images'])
        mask = self.rearrange_features_to_patches(inputs['image_masks'])

        self.visualize_patched_images(images, mask)

        # Get max length across all attention weights
        max_len = max(w.size(-1) for w in attn_weights)

        # Pad each attention weight tensor to max length
        attn_weights = [
            torch.nn.functional.pad(w, (0, pad_len))
            if (pad_len := max_len - w.size(-1)) > 0
            else w for w in attn_weights
        ]

        image_input_idx = inputs['image_input_idx']
        max_len = image_input_idx.max()
        # Stack all padded weights
        attn_weights = reduce(attn_weights, 'b 1 h 1 l -> b l', 'mean')
        max_attn_weight = torch.argsort(attn_weights, dim=-1, descending=True)

        for itoken, idlist in enumerate(max_attn_weight):
            image_patches = []
            for i, idx in enumerate(idlist):
                msel = image_input_idx.squeeze(0) == idx
                if not msel.any():
                    continue
                if not mask[msel].any():
                    continue
                image_patches.append(images[msel].squeeze(0))
                if len(image_patches) > 50:
                    break
            # Create figure and axes grid for image patches
            if image_patches:
                n_patches = len(image_patches)
                n_cols = min(10, n_patches)  # Max 10 columns
                n_rows = (n_patches + n_cols - 1) // n_cols  # Ceiling division

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))

                # Handle single row/column cases
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                # Plot each patch
                for idx, patch in enumerate(image_patches):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].imshow(patch.cpu().numpy())
                    axes[row, col].axis('off')

                # Turn off empty subplots
                for idx in range(n_patches, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.show(block=True)

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
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>",
                                 return_dict_in_generate=True,
                                 output_scores=True, output_logits=False,
                                 output_hidden_states=False, output_attentions=False),
                tokenizer=self.processor.tokenizer
            )

        last_attn_weight = self.collect_attn_weights(output)

        # only get generated tokens; decode them to text
        generated_tokens = output.sequences[0, inputs['input_ids'].size(1):]

        # self.plot_attn_weights(inputs, generated_tokens, last_attn_weight)

        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        topk = [torch.topk(a, k=12, largest=True, sorted=True) for a in output.scores]
        topk_tokens = torch.cat([a.indices for a in topk], dim=0)
        topk_scores = torch.cat([a.values for a in topk], dim=0)
        topk_scores.sigmoid_()

        pw, ph = 60, 15
        total_width = pw * topk_tokens.size(-1)
        total_height = ph * len(topk)

        # Create a new image with the combined width and max height
        combined_img = Image.new('RGBA', (total_width, total_height))

        y_offset = 0
        for alt_token, alt_score in zip(topk_tokens, topk_scores):
            alt_text = self.processor.tokenizer.batch_decode(alt_token.unsqueeze(-1), skip_special_tokens=True)
            # Paste each image into the combined image
            x_offset = 0
            for word, alpha in zip(alt_text, alt_score):
                # Create a new image with red background and alpha channel
                img = Image.new('RGBA', (pw, ph), color=(255, 0, 0, int(alpha * 255)))  # Adjust size as needed
                # Draw the text on the image
                d = ImageDraw.Draw(img)
                d.text((3, 1), word, fill=(255, 255, 255))  # White text
                combined_img.paste(img, (x_offset, y_offset))
                x_offset += img.width
            y_offset += ph

        background = Image.new('RGBA', combined_img.size, (0,0,0))
        alpha_composite = Image.alpha_composite(background, combined_img)
        # Save the combined image
        alpha_composite.save('plot.png')

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
    # image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
    image = Image.open('/home/wenri/Pictures/demoimg.png')
    t0 = time.time()
    generated_text = molmo.generated_kps_points(image,
                                                text='point to the armrest in this image')
    t1 = time.time()
    print(t1 - t0)
    kps, alt = molmo.parse_points_str(generated_text)
    image_with_points = molmo.draw_points(image, kps)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.imshow(image_with_points)
    plt.axis('off')
    plt.show(block=True)
    plt.savefig('plot_with_points.png')


if __name__ == "__main__":
    main()
