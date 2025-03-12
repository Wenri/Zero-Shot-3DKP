import base64
import io

import requests
from openai import OpenAI
from PIL import Image


class GPT4o:
    organization = 'org-G5lzevTDs4hKHMeevxcngnyh'
    project = 'proj_c6daorKHI51Gq4ULtst8FMkT'
    api_key = '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'

    @staticmethod
    def pil_to_base64(pil_image):
        buffered = io.BytesIO()
        # Save the image to the buffer in PNG format
        pil_image.save(buffered, format="PNG")
        # Get the byte data from the buffer
        img_data = buffered.getvalue()
        # Encode the byte data in base64
        base64_encoded = base64.b64encode(img_data).decode('utf-8')
        return base64_encoded

    def __init__(self):
        self.client = OpenAI(organization=self.organization, project=self.project, api_key=self.api_key)

    def get_kplist(self, image):
        image_encoded = self.pil_to_base64(image)
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
                        "text": "List possible salient keypoints (in text)"
                    }
                ]
            },
        ]
        response = self.client.chat.completions.create(
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

    def iter_over_list(self, content):
        for k, v in content.items():
            if isinstance(v, dict):
                yield from self.iter_over_list(v)
            elif isinstance(v, list | set | tuple):
                yield from v
            else:
                yield v


def main():
    image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
    response = GPT4o().get_kplist(image)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
