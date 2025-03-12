import sys
from xml.etree.ElementTree import ParseError

import json
from PIL import Image
from tqdm import tqdm

from data_creation.gpt4o import GPT4o
from data_creation.molmo import Molmo


def main():
    # requests.get("https://picsum.photos/id/237/536/354", stream=True).raw
    filename = '/home/wenri/Data/KeypointNet-All/02691156/1d1244abfefc781f35fc197bbabcd5bd/images/8.png'
    image = Image.open(filename)
    gpt = GPT4o()
    response = gpt.get_kplist(image)
    content = json.loads(response.choices[0].message.content)
    kp_list = [kp for kp in gpt.iter_over_list(content)]

    molmo = Molmo()
    print(','.join(kp_list), flush=True)
    kp_list = tqdm(kp_list)
    for kp in kp_list:
        kp_list.set_description(kp)
        kps = molmo.generated_kps_points(image, text=f"point to the {kp} in this image")
        try:
            kp_list.clear()
            print(kps)
            kps, alt = molmo.parse_points_str(kps)
            if len(kps) >= 5:
                print(f'too many kps for {alt}', file=sys.stderr)
                continue
            print(f"Drawing {alt} in this image", flush=True)
            molmo.draw_points(image, kps)
        except ParseError as e:
            print(e, file=sys.stderr)

    image.show()  # Display the image


if __name__ == '__main__':
    main()
