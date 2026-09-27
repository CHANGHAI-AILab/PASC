"""Red SCC / blue ADC display maps from the segmentation label masks.

    python scripts/00_display_maps.py MASK_DIR OUT_DIR [--ids SLIDE_ID ...]

Reads `MASK_DIR/{slide_id}_labels.png` (0 background, 1 SCC, 2 ADC, 255 no tissue) and writes
`{slide_id}_map.png`. Output PNGs are for figures only.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pasc_consol import masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mask_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--ids', nargs='*', help='slide ids; default = all label masks in mask_dir')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = a.ids or sorted(p.name[:-len('_labels.png')] for p in Path(a.mask_dir).glob('*_labels.png'))
    for sid in ids:
        lab = masks.load_label_png(Path(a.mask_dir) / f'{sid}_labels.png')
        Image.fromarray(masks.colorize(lab)).save(out / f'{sid}_map.png')
    print(f'{len(ids)} maps -> {out}')


if __name__ == '__main__':
    main()
