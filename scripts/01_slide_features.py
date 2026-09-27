"""Slide-level descriptors from the segmentation label masks (Supplementary Methods S2-S3).

    python scripts/01_slide_features.py manifest.csv slide_features.csv --masks MASK_DIR

manifest.csv needs patient_id and slide_id. Each slide is read as
`MASK_DIR/{slide_id}_labels.png` with its pixel size from `MASK_DIR/{slide_id}.json`
(both written by segmentation/assemble_masks.py at the mapping scale); --mpp sets the pixel
size of the label map for slides whose JSON has none.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pasc_consol import descriptors, masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('out_csv')
    ap.add_argument('--masks', required=True, help='folder with {slide_id}_labels.png and {slide_id}.json')
    ap.add_argument('--mpp', type=float, help='default um/px of the label map (x and y)')
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, dtype={'patient_id': str, 'slide_id': str})
    default = (a.mpp, a.mpp) if a.mpp else None
    rows = []
    for r in man.itertuples():
        lab, mpp = masks.load_slide(Path(a.masks) / f'{r.slide_id}_labels.png',
                                    Path(a.masks) / f'{r.slide_id}.json', default)
        rows.append({'patient_id': r.patient_id, 'slide_id': r.slide_id, 'mpp_x': mpp[0], 'mpp_y': mpp[1],
                     **descriptors.slide_descriptors(lab, mpp)})
    out = pd.DataFrame(rows)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print(f'{len(out)} slides, {int(out.eligible.sum())} eligible -> {a.out_csv}')


if __name__ == '__main__':
    main()
