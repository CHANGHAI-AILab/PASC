"""Label coding and post-processing of the tissue maps (Supplementary Method S2).

Labels: 0 non-tumor/background, 1 squamous carcinoma (SCC), 2 adenocarcinoma (ADC),
255 ignore. In annotations 255 marks folds, severe artifact and uncertain necrosis; in the
whole-slide label maps it marks mapping-scale blocks without sampled tissue.
"""
import cv2
import numpy as np

BACKGROUND, SCC, ADC, IGNORE = 0, 1, 2, 255
CLASS_NAMES = ('background', 'SCC', 'ADC')
MIN_PROB = 0.5
CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSING_HALO = 2      # rows a 3 x 3 closing (dilation, then erosion) reads beyond a band
MAPPING_SCALE = 40    # level-0 pixels per mapping-scale pixel, per axis


def labels_from_probs(probs, min_prob=MIN_PROB):
    """Softmax probabilities (C, H, W) -> uint8 labels.

    The argmax class is kept; pixels whose maximum probability is below min_prob are
    background. A pixel therefore gets a single label when SCC and ADC compete.
    """
    probs = np.asarray(probs)
    lab = probs.argmax(axis=0).astype(np.uint8)
    lab[probs.max(axis=0) < min_prob] = BACKGROUND
    return lab


def close_labels(lab):
    """One 3 x 3 elliptical closing of the SCC and ADC maps.

    Closing only fills background pixels; where both classes would fill the same pixel it
    becomes SCC.
    """
    out = lab.copy()
    for c in (ADC, SCC):  # SCC last, so it wins where both fill
        closed = cv2.morphologyEx((lab == c).astype(np.uint8), cv2.MORPH_CLOSE, CLOSING_KERNEL) > 0
        out[closed & (lab == BACKGROUND)] = c
    return out


def _blocks(a, factor, fill):
    h, w = a.shape
    oh, ow = -(-h // factor), -(-w // factor)
    padded = np.full((oh * factor, ow * factor), fill, a.dtype)
    padded[:h, :w] = a
    return padded.reshape(oh, factor, ow, factor)


def reduce_labels(lab, factor=MAPPING_SCALE, tissue=None):
    """Level-0 labels -> mapping scale, one pixel per factor x factor block.

    A block is SCC if it contains any SCC pixel, otherwise ADC if it contains any ADC pixel,
    otherwise ignore if all its pixels are ignore, otherwise background. With a level-0
    tissue mask, background blocks without any tissue pixel become ignore, so the non-ignore
    pixels of the result are the sampled tissue. Partial blocks at the right and lower
    borders are kept.
    """
    blocks = _blocks(np.asarray(lab, np.uint8), factor, IGNORE)
    out = np.where((blocks == IGNORE).all(axis=(1, 3)), IGNORE, BACKGROUND).astype(np.uint8)
    if tissue is not None:
        out[~_blocks(np.asarray(tissue, bool), factor, False).any(axis=(1, 3))] = IGNORE
    out[(blocks == ADC).any(axis=(1, 3))] = ADC
    out[(blocks == SCC).any(axis=(1, 3))] = SCC
    return out
