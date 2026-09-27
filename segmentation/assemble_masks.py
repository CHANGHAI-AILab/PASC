"""Assemble per-tile predictions into whole-slide label masks.

    python assemble_masks.py predictions/ masks/

For each slide:
  1. Each tile's float16 probabilities are converted to labels: the argmax class, or
     background where the maximum softmax probability is below 0.5 (labels.labels_from_probs).
  2. Labels are placed at their level-0 coordinates. Tiles lie on a 256-pixel grid; the last
     tile in each row and column was shifted inward to end at the border and overlaps its
     neighbour. Every coordinate is written once: overlapping pixels keep the grid tile's
     label, and the shifted tile contributes only the pixels beyond the grid. No averaging
     or voting is done. White padding (slides smaller than a tile) is cropped. Skipped tiles
     (< 20% tissue) are background. The tile tissue masks (gray value < 220) are placed the
     same way; skipped tiles and padding are not tissue.
  3. One 3 x 3 elliptical closing is applied at level 0 (labels.close_labels).
  4. The level-0 map is reduced 40x per axis to the mapping scale (labels.reduce_labels).
     Background blocks without any tissue pixel become 255, so the non-ignore pixels of the
     label map are the sampled tissue.
Steps 2-4 run on row bands of 2,560 level-0 rows (with a 2-row halo for the closing), so a
whole level-0 slide never has to be held in memory; the result equals whole-slide processing.

Output: masks/{slide}_labels.png, a lossless uint8 PNG (0 background tissue, 1 SCC, 2 ADC,
255 no sampled tissue) at the mapping scale, used for quantification, and masks/{slide}.json
with the level-0 size, the level-0 mpp and the mapping-scale mpp (level-0 mpp x 40).
"""
import argparse
import json
import os
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image

from labels import CLOSING_HALO, MAPPING_SCALE, close_labels, labels_from_probs, reduce_labels

Image.MAX_IMAGE_PIXELS = None
TILE_RE = re.compile(r'\[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]')
BAND = 2560  # multiple of the 256-px tile and of the 40x mapping scale


def list_tiles(pred_dir):
    tiles = []
    for name in sorted(os.listdir(pred_dir)):
        m = TILE_RE.search(name)
        if m and name.endswith('.npz'):
            tiles.append((name,) + tuple(int(v) for v in m.groups()))
    return tiles


def _own_start(pos, extent, size):
    """First coordinate a tile writes: its origin on the grid, else the end of the grid."""
    if pos % size == 0:
        return pos
    return (extent // size) * size


def _tile_labels(pred_dir, name):
    with np.load(os.path.join(pred_dir, name)) as f:
        if 'tissue' not in f:
            raise ValueError('%s has no tissue mask; re-run predict_tiles.py' % name)
        return labels_from_probs(f['probs'].astype(np.float32)), f['tissue'].astype(bool)


def place(pred_dir, tiles, width, height, top, bottom):
    """Rows top..bottom-1 of the level-0 label map (before closing) and tissue mask."""
    lab = np.zeros((bottom - top, width), np.uint8)
    tissue = np.zeros((bottom - top, width), bool)
    for name, x, y, w, h in tiles:
        x0, y0 = _own_start(x, width, w), _own_start(y, height, h)
        x1, y1 = min(x + w, width), min(y + h, height)
        r0, r1 = max(y0, top), min(y1, bottom)
        if x1 <= x0 or r1 <= r0:
            continue
        tile, tile_tissue = _tile_labels(pred_dir, name)
        lab[r0 - top:r1 - top, x0:x1] = tile[r0 - y:r1 - y, x0 - x:x1 - x]
        tissue[r0 - top:r1 - top, x0:x1] = tile_tissue[r0 - y:r1 - y, x0 - x:x1 - x]
    return lab, tissue


def assemble_bands(pred_dir, width, height, closing=True, band=BAND):
    """Yield (row offset, closed level-0 label band, tissue band) for the whole slide."""
    tiles = list_tiles(pred_dir)
    for r0 in range(0, height, band):
        r1 = min(r0 + band, height)
        top, bottom = max(r0 - CLOSING_HALO, 0), min(r1 + CLOSING_HALO, height)
        near = [t for t in tiles if t[2] < bottom and t[2] + t[4] > top]
        lab, tissue = place(pred_dir, near, width, height, top, bottom)
        if closing:
            lab = close_labels(lab)
        yield r0, lab[r0 - top:r1 - top], tissue[r0 - top:r1 - top]


def assemble(pred_dir, width, height, closing=True, factor=MAPPING_SCALE, band=BAND):
    """Mapping-scale uint8 label map of one slide."""
    if band % factor:
        raise ValueError('band height must be a multiple of the mapping scale')
    return np.vstack([reduce_labels(lab, factor, tissue)
                      for _, lab, tissue in assemble_bands(pred_dir, width, height, closing, band)])


def save_png(lab, path):
    tmp = path + '.part'
    Image.fromarray(np.ascontiguousarray(lab), mode='L').save(tmp, format='PNG')
    os.replace(tmp, path)


def run_one(job):
    slide, pred_dir, out_dir, default_mpp = job
    try:
        with open(os.path.join(pred_dir, 'slide.json')) as fh:
            info = json.load(fh)
        mpp_x, mpp_y = info.get('mpp_x') or default_mpp, info.get('mpp_y') or default_mpp
        if not (mpp_x and mpp_y):
            raise ValueError('no mpp in slide.json; pass --mpp')
        lab = assemble(pred_dir, info['width'], info['height'])
        save_png(lab, os.path.join(out_dir, slide + '_labels.png'))
        meta = {'slide': slide, 'width': info['width'], 'height': info['height'],
                'level0_mpp_x': mpp_x, 'level0_mpp_y': mpp_y, 'scale': MAPPING_SCALE,
                'mpp_x': mpp_x * MAPPING_SCALE, 'mpp_y': mpp_y * MAPPING_SCALE}
        with open(os.path.join(out_dir, slide + '.json'), 'w') as fh:
            json.dump(meta, fh, indent=1)
        print(slide)
    except Exception as e:
        print('%s: %s' % (slide, e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pred_dir', help='folder with one sub-folder of tile predictions per slide')
    ap.add_argument('out_dir')
    ap.add_argument('--mpp', type=float, default=None, help='level-0 um/px for slides whose slide.json has none')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(s, os.path.join(args.pred_dir, s), args.out_dir, args.mpp)
            for s in sorted(os.listdir(args.pred_dir)) if os.path.isdir(os.path.join(args.pred_dir, s))]
    with Pool(args.workers) as pool:
        pool.map(run_one, jobs)


if __name__ == '__main__':
    main()
