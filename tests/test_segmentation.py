import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('cv2')
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'segmentation'))
import assemble_masks as am  # noqa: E402
import labels  # noqa: E402
import slice_wsi  # noqa: E402


def probs_for(lab, p=0.9):
    """One-hot-ish probabilities (3, H, W) with max probability p at the given labels."""
    lab = np.asarray(lab)
    probs = np.full((3,) + lab.shape, (1 - p) / 2, np.float32)
    for c in range(3):
        probs[c][lab == c] = p
    return probs


def test_softmax_threshold():
    probs = np.array([[[0.2]], [[0.45]], [[0.35]]], np.float32)   # max 0.45 < 0.5 -> background
    assert labels.labels_from_probs(probs).tolist() == [[0]]
    probs = np.array([[[0.1]], [[0.3]], [[0.6]]], np.float32)
    assert labels.labels_from_probs(probs).tolist() == [[labels.ADC]]


def test_tile_grid_stride_and_border():
    origins = slice_wsi.tile_origins(600, 300)
    xs = sorted({x for x, _ in origins})
    ys = sorted({y for _, y in origins})
    assert xs == [0, 256, 600 - 256] and ys == [0, 300 - 256]
    assert slice_wsi.tile_origins(100, 100) == [(0, 0)]
    assert slice_wsi.tile_name('S1', 256, 0) == 'S1 [x=256,y=0,w=256,h=256].tif'
    assert slice_wsi.clean_name('12 3-A (1)') == '123A1'


def test_tissue_filter_and_padding():
    white = np.full((256, 256, 3), 255, np.uint8)
    tile = white.copy()
    tile[:51] = 150                       # 51 / 256 rows = 19.9% tissue
    assert slice_wsi.tissue_fraction(tile) < slice_wsi.MIN_TISSUE
    tile[:52] = 150
    assert slice_wsi.tissue_fraction(tile) >= slice_wsi.MIN_TISSUE
    padded = slice_wsi.pad_white(np.zeros((100, 120, 3), np.uint8))
    assert padded.shape == (256, 256, 3) and padded[:100, :120].max() == 0 and padded[100:].min() == 255


def test_reduce_labels_priority():
    lab = np.zeros((80, 100), np.uint8)
    lab[0, 0] = labels.ADC
    lab[1, 1] = labels.SCC                # block (0, 0): SCC wins over ADC
    lab[45, 45] = labels.ADC              # block (1, 1): ADC
    lab[:40, 40:80] = labels.IGNORE       # block (0, 1): all ignore
    lab[79, 99] = labels.SCC              # partial block at the right border
    small = labels.reduce_labels(lab, 40)
    assert small.tolist() == [[labels.SCC, labels.IGNORE, 0], [0, labels.ADC, labels.SCC]]


def test_closing_fills_one_pixel_gaps():
    lab = np.zeros((20, 20), np.uint8)
    lab[5:15, 5:9] = labels.SCC
    lab[5:15, 10:14] = labels.SCC          # one-pixel gap at column 9
    lab[0, 0] = labels.ADC
    closed = labels.close_labels(lab)
    assert (closed[6:14, 9] == labels.SCC).all()
    assert closed[0, 0] == labels.ADC and closed[18, 18] == 0


def save_tile(path, x, y, lab, tissue=None):
    tissue = np.ones((256, 256), bool) if tissue is None else tissue
    np.savez_compressed(path / (slice_wsi.tile_name('S', x, y)[:-4] + '.npz'),
                        probs=probs_for(lab).astype(np.float16), tissue=tissue)


def level0(tmp_path, width, height, **kw):
    """Level-0 labels via the banded assembly (factor 1 = no reduction)."""
    return am.assemble(str(tmp_path), width, height, factor=1, **kw)


def test_assemble_overlap_written_once(tmp_path):
    # 300 x 256 slide: grid tile at x = 0 and a border tile at x = 44 overlapping it
    save_tile(tmp_path, 0, 0, np.full((256, 256), labels.SCC))
    save_tile(tmp_path, 300 - 256, 0, np.full((256, 256), labels.ADC))
    lab = level0(tmp_path, 300, 256, closing=False)
    assert lab.shape == (256, 300) and lab.dtype == np.uint8
    assert (lab[:, :256] == labels.SCC).all() and (lab[:, 256:] == labels.ADC).all()


def test_assemble_crops_padding_and_skipped_tiles(tmp_path):
    # slide narrower than a tile (padding cropped); second tile row missing (skipped, < 20% tissue)
    lab = np.zeros((256, 256))
    lab[:50, :50] = labels.SCC
    save_tile(tmp_path, 0, 0, lab)
    out = level0(tmp_path, 200, 400, closing=False)
    assert out.shape == (400, 200)
    assert (out == labels.SCC).sum() == 2500 and (out[:256] != labels.IGNORE).all()
    assert (out[256:] == labels.IGNORE).all()   # skipped tile: no sampled tissue


def test_non_tissue_blocks_are_ignore(tmp_path):
    lab = np.zeros((256, 256))
    lab[:10, 200:210] = labels.SCC            # tumor in a non-tissue area is kept
    tissue = np.zeros((256, 256), bool)
    tissue[50, 50] = True                     # one tissue pixel makes its block tissue
    save_tile(tmp_path, 0, 0, lab, tissue)
    out = am.assemble(str(tmp_path), 256, 256, closing=False)
    assert out.shape == (7, 7)
    assert out[1, 1] == 0 and out[0, 5] == labels.SCC
    assert (out != labels.IGNORE).sum() == 2
    assert slice_wsi.tissue_mask(np.array([[[219, 219, 219], [220, 220, 220]]], np.uint8)).tolist() == [[True, False]]


def test_banded_equals_whole_slide(tmp_path):
    rng = np.random.default_rng(0)
    width, height = 600, 700
    for x, y in slice_wsi.tile_origins(width, height):
        tile = rng.choice([0, 1, 2], size=(256, 256), p=[0.6, 0.2, 0.2])
        tile[:, 128:] = 0                     # right half background, sparse tissue
        save_tile(tmp_path, x, y, tile, rng.random((256, 256)) < 0.0003)
    whole, tissue = am.place(str(tmp_path), am.list_tiles(str(tmp_path)), width, height, 0, height)
    expect = labels.reduce_labels(labels.close_labels(whole), 40, tissue)
    got = am.assemble(str(tmp_path), width, height, band=80)
    assert (got == expect).all() and (got == labels.IGNORE).any()
    assert (level0(tmp_path, width, height, band=80) == labels.reduce_labels(labels.close_labels(whole), 1, tissue)).all()


def test_label_png_roundtrip(tmp_path):
    lab = np.array([[0, 1], [2, 255]], np.uint8)
    am.save_png(lab, str(tmp_path / 'S_labels.png'))
    assert (np.array(Image.open(tmp_path / 'S_labels.png')) == lab).all()
