"""Convert labelme polygon annotations (*.json) to class-index label PNGs.

    python json_to_dataset.py annotations/ dataset/ --classes _background_ SCC ADC --ignore ignore

Each JSON (one 256 x 256 level-0 patch) is rasterized with labelme; shape labels are mapped
to their index in --classes, and labels listed in --ignore (folds, severe artifact,
uncertain necrosis) to 255. The image is written to dataset/Images/{name}.tif and the label
(uint8 PNG) to dataset/Labels/{name}.png. Annotations were made with labelme 5.2.1 (labelme.utils.shapes_to_label).
"""
import argparse
import base64
import json
import os

import numpy as np
from PIL import Image

IGNORE = 255
PATCH = 256


def main():
    from labelme import utils

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json_dir')
    ap.add_argument('out')
    ap.add_argument('--classes', nargs='+', required=True, help='class names, background first')
    ap.add_argument('--ignore', nargs='*', default=['ignore'], help='labels written as 255')
    args = ap.parse_args()

    img_dir = os.path.join(args.out, 'Images')
    lbl_dir = os.path.join(args.out, 'Labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.json_dir)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(args.json_dir, fname)
        try:
            data = json.load(open(path))
            image_data = data.get('imageData')
            if not image_data:
                with open(os.path.join(os.path.dirname(path), data['imagePath']), 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            img = utils.img_b64_to_arr(image_data)
            if img.shape[:2] != (PATCH, PATCH):
                print('%s: patch is %s, expected %d x %d' % (path, img.shape[:2], PATCH, PATCH))

            name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                name_to_value.setdefault(shape['label'], len(name_to_value))
            lbl = utils.shapes_to_label(img.shape, data['shapes'], name_to_value)
            if isinstance(lbl, tuple):  # labelme >= 4 returns (cls, ins)
                lbl = lbl[0]

            new = np.zeros(img.shape[:2], np.uint8)
            for name, value in name_to_value.items():
                if name in args.ignore:
                    new[np.asarray(lbl) == value] = IGNORE
                elif name in args.classes:
                    new[np.asarray(lbl) == value] = args.classes.index(name)
                else:
                    raise ValueError('label %r not in --classes or --ignore' % name)

            stem = os.path.splitext(fname)[0]
            Image.fromarray(img).save(os.path.join(img_dir, stem + '.tif'))
            Image.fromarray(new, mode='L').save(os.path.join(lbl_dir, stem + '.png'))
        except Exception as e:
            print('%s: %s' % (path, e))


if __name__ == '__main__':
    main()
