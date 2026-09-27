"""Slide- and patient-level spatial descriptors (Supplementary Method S3, Table S3).

Slide descriptors are computed on the mapping-scale label map (segmentation/assemble_masks.py); all
lengths and areas are in its pixels, except the tumor area used for eligibility, which is in
square micrometers. Patient aggregation gives C_raw, D1-D5, B, S1-S3 and V.
"""
import cv2
import numpy as np
from scipy import ndimage

from .constants import (CANNY_THRESHOLDS, FRONT_THICKNESS_PX, INTERFACE_DILATION_RADIUS,
                        LABEL_ADC, LABEL_IGNORE, LABEL_SCC, LOGIT_EPS, MIN_REGION_PIXELS,
                        MIN_TUMOR_AREA_UM2, SHAPE_CLOSING_KERNEL, SIGNIFICANT_REGION_FRACTION)

_EIGHT = np.ones((3, 3), dtype=int)


def region_sizes(binary):
    """Pixel counts of 8-connected components, largest first."""
    lab, n = ndimage.label(binary, structure=_EIGHT)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.bincount(lab.ravel())[1:])[::-1]


def _edges(binary):
    return cv2.Canny(binary.astype(np.uint8) * 255, *CANNY_THRESHOLDS) > 0


def _shape(scc, a_scc):
    """Isoperimetric shape factor P^2 / (4 pi A) and P / sqrt(A) after one 3x3 closing."""
    closed = cv2.morphologyEx(scc.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones(SHAPE_CLOSING_KERNEL, np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0
    p = sum(cv2.arcLength(c, True) for c in contours)
    return p ** 2 / (4 * np.pi * a_scc), p / np.sqrt(a_scc)


def _interface_and_front(scc, adc):
    r = INTERFACE_DILATION_RADIUS
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    adc_dil = cv2.dilate(adc.astype(np.uint8), ker) > 0
    scc_dil = cv2.dilate(scc.astype(np.uint8), ker) > 0
    e_scc, e_adc = _edges(scc), _edges(adc)
    n_scc, n_adc = int(e_scc.sum()), int(e_adc.sum())
    hit_scc = int((e_scc & adc_dil).sum())
    hit_adc = int((e_adc & scc_dil).sum())
    interface = hit_scc / n_scc if n_scc else 0.0
    interface_sym = (hit_scc + hit_adc) / (n_scc + n_adc) if n_scc + n_adc else 0.0

    dist = ndimage.distance_transform_edt(scc | adc)
    front = (dist > 0) & (dist <= FRONT_THICKNESS_PX)
    d_scc = dist[scc]
    return interface, interface_sym, float((scc & front).sum() / scc.sum()), float(d_scc.mean())


def slide_descriptors(lab, mpp):
    """Descriptors of one mapping-scale label map (0 background, 1 SCC, 2 ADC, 255 ignore).

    `mpp` = (mpp_x, mpp_y) of `lab`'s pixels. N_px, the sampled tissue, is the number of
    non-ignore pixels (blocks without tissue are 255 in the label map). A slide is eligible when its tumor area
    (SCC + ADC pixels x mpp_x x mpp_y) is >= 1.2 x 10^6 um^2.
    CCCount and DCR use regions >= 10 px; the DCR denominator is all SCC pixels.
    Significant-region count and second-largest fraction use all regions. A slide is
    multifocal when it has >= 2 significant regions or a second region >= 5% of SCC.
    A slide without SCC keeps its tumor area and gets 0 for every SCC descriptor.
    """
    scc = lab == LABEL_SCC
    adc = lab == LABEL_ADC
    n_px = int((lab != LABEL_IGNORE).sum())
    a_scc, a_adc = int(scc.sum()), int(adc.sum())
    t_um2 = (a_scc + a_adc) * float(mpp[0]) * float(mpp[1])
    d = {
        'N_px': n_px, 'A_SCC_px': a_scc, 'A_ADC_px': a_adc, 'T_px': a_scc + a_adc, 'T_um2': t_um2,
        'eligible': t_um2 >= MIN_TUMOR_AREA_UM2,
        'CCCount': 0, 'LargestRegion_px': 0, 'DCR': 0.0,
        'LesionCount': 0, 'SecondClusterFrac': 0.0, 'Multifocal': 0,
        'ShapeFactor': 0.0, 'PerimeterAreaRatio': 0.0,
        'InterfaceFrac': 0.0, 'InterfaceFracSym': 0.0, 'FrontFrac': 0.0, 'FrontDistance_px': 0.0,
    }
    if a_scc == 0:
        return d

    sizes = region_sizes(scc)
    kept = sizes[sizes >= MIN_REGION_PIXELS]
    n_sig = int((sizes >= SIGNIFICANT_REGION_FRACTION * a_scc).sum())
    second = sizes[1] / a_scc if sizes.size > 1 else 0.0
    d.update(LesionCount=n_sig, SecondClusterFrac=float(second),
             Multifocal=int(n_sig >= 2 or second >= SIGNIFICANT_REGION_FRACTION),
             CCCount=int(kept.size))
    if kept.size:
        d.update(LargestRegion_px=int(kept[0]), DCR=float(kept[0] / a_scc))
        d['ShapeFactor'], d['PerimeterAreaRatio'] = _shape(scc, a_scc)
    (d['InterfaceFrac'], d['InterfaceFracSym'],
     d['FrontFrac'], d['FrontDistance_px']) = _interface_and_front(scc, adc)
    return d


def _logit(p):
    p = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return float(np.log(p / (1 - p)))


def patient_descriptors(slides):
    """Aggregate slide dicts across eligible slides (tumor area >= 1.2 x 10^6 um^2).

    Proportions and shape/interface/front measures are tumor-area (T_px) weighted means;
    count and extremum measures use the slide maximum. V_raw = logit(weighted mean of mapped
    tumor / sampled tissue, T_px / N_px).
    """
    el = [s for s in slides if s['eligible']]
    if not el:
        raise ValueError('patient has no eligible slide')
    t = np.array([s['T_px'] for s in el], float)

    def wmean(key):
        return float(np.average([s[key] for s in el], weights=t))

    frac = sum(s['A_SCC_px'] for s in el) / t.sum()
    dcr = wmean('DCR')
    shape = wmean('ShapeFactor')
    occupancy = float(np.average(t / np.array([s['N_px'] for s in el], float), weights=t))
    return {
        'n_eligible_slides': len(el),
        'SCC_in_tumor_pt': frac,
        'C_raw': _logit(frac),
        'DCR_wavg': dcr,
        'D1_raw': 1 - dcr,
        'D2_raw': int(any(s['Multifocal'] for s in el)),
        'D3_raw': float(np.log1p(max(s['CCCount'] for s in el))),
        'D4_raw': float(np.log1p(max(s['LesionCount'] for s in el))),
        'D5_raw': float(max(s['SecondClusterFrac'] for s in el)),
        'B_raw': float(np.log(shape)) if shape > 0 else np.nan,
        'S1_raw': wmean('InterfaceFracSym'),
        'S2_raw': wmean('FrontFrac'),
        'S3_raw': -float(np.log1p(wmean('FrontDistance_px'))),
        'V_raw': _logit(occupancy),
    }

