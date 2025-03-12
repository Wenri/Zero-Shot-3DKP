import base64
import io
import json

import requests
from openai import OpenAI
from PIL import Image

from data_creation.molmo import Molmo


def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    # Save the image to the buffer in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_data = buffered.getvalue()
    # Encode the byte data in base64
    base64_encoded = base64.b64encode(img_data).decode('utf-8')
    return base64_encoded


def get_kplist(image):
    client = OpenAI(organization='org-G5lzevTDs4hKHMeevxcngnyh',
                    project='proj_c6daorKHI51Gq4ULtst8FMkT',
                    api_key='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')
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
                    "text": "Point to corner of the back of the chair in this image. Output the point location in the image (xy) coordinate system"
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


def norm_location_list(content, name=None):
    for partname, location in content.items():
        if isinstance(location, list):
            yield from enumerate(location)
        elif isinstance(location, dict):
            yield from norm_location_list(location, partname)
        else:
            yield name, content
            break


def main():
    image = Image.open('/home/wenri/pCloudDrive/Screenshots/ChatGPT/snapshot02 (s).png')
    w, h = image.width, image.height
    response = get_kplist(image)
    content = json.loads(response.choices[0].message.content)
    kps = {k: v for k, v in norm_location_list(content)}
    print(kps)
    Molmo.draw_points(image, kps, kps_wh=(w, h))

    image.show()  # Display the image


if __name__ == "__main__":
    main()