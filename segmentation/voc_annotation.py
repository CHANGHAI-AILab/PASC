"""Write the train / held-out split lists for the segmentation dataset.

    python voc_annotation.py dataset/

The split is made at the whole-slide level: 80% of the WSIs (random.seed(0)) go to
ImageSets/Segmentation/train.txt (development) and the remaining 20% to val.txt (held out
for technical quality control). A patch belongs to the WSI named before ' [' in its file
name (`{slide} [x=X,y=Y,w=256,h=256]`). The pixel values of all labels are then counted as
a format check (0 background, 1 SCC, 2 ADC, 255 ignore).
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def slide_of(name):
    return name.split(' [')[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--dev-fraction', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    segfilepath = os.path.join(args.dataset, 'Labels')
    save_base = os.path.join(args.dataset, 'ImageSets/Segmentation')
    os.makedirs(save_base, exist_ok=True)

    names = sorted(s[:-4] for s in os.listdir(segfilepath) if s.endswith('.png'))
    slides = sorted({slide_of(n) for n in names})
    dev = set(random.sample(slides, int(round(len(slides) * args.dev_fraction))))
    train = [n for n in names if slide_of(n) in dev]
    val = [n for n in names if slide_of(n) not in dev]

    for fname, lines in (('trainval.txt', names), ('train.txt', train), ('val.txt', val)):
        with open(os.path.join(save_base, fname), 'w') as fh:
            fh.writelines(n + '\n' for n in lines)
    print('WSIs: %d development, %d held out; patches: %d train, %d val'
          % (len(dev), len(slides) - len(dev), len(train), len(val)))

    classes_nums = np.zeros([256], np.int64)
    for name in tqdm(names):
        png = np.array(Image.open(os.path.join(segfilepath, name + '.png')), np.uint8)
        if png.ndim > 2:
            print('%s has shape %s; labels must be single-channel class indices.' % (name, png.shape))
        classes_nums += np.bincount(png.reshape(-1), minlength=256)
    for v in np.nonzero(classes_nums)[0]:
        print('value %3d: %d px' % (v, classes_nums[v]))
    unexpected = set(np.nonzero(classes_nums)[0].tolist()) - {0, 1, 2, 255}
    if unexpected:
        print('Unexpected label values %s; use 0 background, 1 SCC, 2 ADC, 255 ignore.' % sorted(unexpected))


if __name__ == '__main__':
    main()
