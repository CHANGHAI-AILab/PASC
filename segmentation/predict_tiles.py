"""Predict every tile of every slide with the trained DeepLab-v3+ model.

    python predict_tiles.py --weights logs/best_epoch_weights.pth --tiles tiles/ --out predictions/

For each tile the softmax probabilities (3 x 256 x 256: background, SCC, ADC) are saved as
float16 in {out}/{slide}/{tile}.npz (key `probs`), with the tile's tissue mask (key `tissue`,
gray value < 220, as in slice_wsi.py); slide.json is copied alongside. Tiles that
already have a prediction are skipped. --range i:j restricts the run to slides i..j-1 of the
sorted slide list, which lets several GPUs share one tile folder.
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from slice_wsi import tissue_mask

IMG_EXT = ('.png', '.tif', '.tiff')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='model state dict, e.g. best_epoch_weights.pth')
    ap.add_argument('--tiles', required=True, help='folder with one sub-folder of tiles per slide')
    ap.add_argument('--out', required=True)
    ap.add_argument('--range', default=None, help='slide index range i:j')
    ap.add_argument('--num-classes', type=int, default=3)
    ap.add_argument('--backbone', default='xception')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    slides = sorted(d for d in os.listdir(args.tiles) if os.path.isdir(os.path.join(args.tiles, d)))
    if args.range:
        i, j = (int(v) for v in args.range.split(':'))
        slides = slides[i:j]
    deeplab = DeeplabV3(args.weights, num_classes=args.num_classes, backbone=args.backbone, cuda=not args.cpu)

    for slide in slides:
        src = os.path.join(args.tiles, slide)
        dst = os.path.join(args.out, slide)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, 'slide.json'), dst)
        done = set(os.listdir(dst))
        names = sorted(n for n in os.listdir(src)
                       if n.lower().endswith(IMG_EXT) and os.path.splitext(n)[0] + '.npz' not in done)
        for name in tqdm(names, desc=slide):
            image = Image.open(os.path.join(src, name)).convert('RGB')
            probs = deeplab.predict_probs(image)
            np.savez_compressed(os.path.join(dst, os.path.splitext(name)[0] + '.npz'),
                                probs=probs.astype(np.float16), tissue=tissue_mask(image))


if __name__ == '__main__':
    main()
