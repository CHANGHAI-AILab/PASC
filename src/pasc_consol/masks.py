"""Tissue-map I/O (Supplementary Method S2).

segmentation/assemble_masks.py writes one lossless uint8 label PNG per slide,
`{slide}_labels.png` (0 non-tumor tissue/background, 1 squamous carcinoma (SCC), 2 adenocarcinoma
(ADC), 255 ignore: no sampled tissue), at the mapping scale (level 0 reduced 40x per axis), and `{slide}.json`
with its pixel size (mpp_x, mpp_y), used to convert pixel counts to physical area.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import LABEL_ADC, LABEL_BACKGROUND, LABEL_IGNORE, LABEL_SCC

Image.MAX_IMAGE_PIXELS = None


def load_label_png(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path)
    if img.mode not in ('L', 'P'):
        raise ValueError(f'{path.name}: expected single-channel uint8 label map, got mode {img.mode}')
    lab = np.asarray(img, dtype=np.uint8)
    bad = set(np.unique(lab).tolist()) - {LABEL_BACKGROUND, LABEL_SCC, LABEL_ADC, LABEL_IGNORE}
    if bad:
        raise ValueError(f'{path.name}: unexpected label values {sorted(bad)}')
    return lab


def load_mpp(path, default=None):
    """(mpp_x, mpp_y) of the label map from the slide JSON; `default` when missing."""
    path = Path(path)
    if path.exists():
        info = json.loads(path.read_text())
        if info.get('mpp_x') and info.get('mpp_y'):
            return float(info['mpp_x']), float(info['mpp_y'])
    if default is None:
        raise ValueError(f'{path}: no mpp_x / mpp_y; pass a default pixel size')
    return default


def load_slide(label_png, slide_json, default_mpp=None):
    """Label map of one slide and its pixel size (mpp_x, mpp_y) in micrometers."""
    return load_label_png(label_png), load_mpp(slide_json, default_mpp)


def colorize(lab, scc_rgb=(205, 40, 40), adc_rgb=(40, 110, 190), background=(255, 255, 255)):
    """Display colors used in Fig. 1b / Fig. 3a / Fig. S1: red SCC, blue ADC."""
    rgb = np.empty(lab.shape + (3,), np.uint8)
    rgb[:] = background
    rgb[lab == LABEL_SCC] = scc_rgb
    rgb[lab == LABEL_ADC] = adc_rgb
    return rgb
