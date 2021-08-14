import os
from argparse import ArgumentParser
import cv2
from PIL import Image

parser = ArgumentParser()
parser.add_argument('--img_dirs', type=str, help='img_address', default='.')
parser.add_argument('--what2what', type=str, help='what type to what type', default='png2jpg')
args = parser.parse_args()

img_dirs = args.img_dirs
what2what = args.what2what

first, second = what2what.split('2')
for img_name in os.listdir(img_dirs):
    if img_name.endswith(first):
        base_name = img_name[:-4]
        img_address = os.path.join(img_dirs, img_name)
        # img = Image.open(img_address)
        # rgb_im = img.convert('RGB')
        # rgb_im.save(os.path.join(img_dirs, base_name + f'.{second}'))
        img = cv2.imread(img_address)
        file_name = os.path.join(img_dirs, base_name + f'.{second}')
        cv2.imwrite(file_name, img)
        print(file_name)
