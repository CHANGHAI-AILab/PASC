"""Cut whole-slide images into the 256 x 256 level-0 tiles used for segmentation.

    python slice_wsi.py /path/wsi tiles/

Slides are 40x scans; tiles are read at level 0 (0.25 um/px) without resizing, on a
256-pixel grid (stride 256). The last tile in each row and column is moved inward so that
it ends at the image border, and therefore overlaps its neighbour. A slide smaller than one
tile is padded with white, and areas outside the scanned region are read as white. Tiles
with less than 20% tissue are skipped; a tissue pixel has a gray value below 220.

Output per slide:
    {out_dir}/{slide}/{slide} [x=X,y=Y,w=256,h=256].tif   X, Y in level-0 pixels
    {out_dir}/{slide}/slide.json                            level-0 size, mpp, tile counts
`slide` is the file stem with all characters other than letters, digits and CJK removed.

Backends:
  openslide  any format OpenSlide reads (NDPI, SVS, MRXS); level-0 read_region.
  ndpisplit  NDPI files; `ndpisplit -Ex{mag},z0,x,y,w,h` at the native magnification.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TILE = 256
TISSUE_GRAY = 220
MIN_TISSUE = 0.2
WSI_EXT = ('.ndpi', '.svs', '.mrxs', '.tif', '.tiff')


def clean_name(stem):
    return re.sub('[^\\u4e00-\\u9fa5a-zA-Z0-9]', '', stem)


def tile_name(slide, x, y, size=TILE):
    return '%s [x=%d,y=%d,w=%d,h=%d].tif' % (slide, x, y, size, size)


def _axis_origins(n, size):
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, size))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def tile_origins(w, h, size=TILE):
    """Tile corners on a stride-`size` grid; the last tile in each axis ends at the border."""
    return [(x, y) for y in _axis_origins(h, size) for x in _axis_origins(w, size)]


def pad_white(region, size=TILE):
    out = np.full((size, size, 3), 255, np.uint8)
    region = np.asarray(region)[:size, :size, :3]
    out[:region.shape[0], :region.shape[1]] = region
    return out


def tissue_mask(rgb):
    """Tissue pixels of an RGB tile: gray value (0.299 R + 0.587 G + 0.114 B) below 220."""
    gray = np.asarray(rgb, np.float32)[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    return gray < TISSUE_GRAY


def tissue_fraction(rgb):
    return float(tissue_mask(rgb).mean())


def _mpp_from_tiff(tags):
    res = tags.get('XResolution'), tags.get('YResolution')
    unit = tags.get('ResolutionUnit')
    unit = getattr(unit, 'value', unit)
    per = {2: 25400.0, 3: 10000.0}.get(int(unit) if unit is not None else 0)
    if per is None or None in res:
        return None, None
    return tuple(per * r[1] / r[0] if r[0] else None for r in res)


def slide_info(path, backend):
    """Level-0 width, height, objective magnification and mpp (x, y)."""
    if backend == 'ndpisplit':
        import tifffile
        with tifffile.TiffFile(path) as f:
            page = f.pages[0]
            tags = {t.name: t.value for t in page.tags.values()}
            mag = next((t.value for t in page.tags.values() if '65421' in t.name), 0)
        mpp_x, mpp_y = _mpp_from_tiff(tags)
        return int(tags['ImageWidth']), int(tags['ImageLength']), int(mag), mpp_x, mpp_y
    import openslide
    wsi = openslide.OpenSlide(path)
    w, h = wsi.level_dimensions[0]
    p = wsi.properties
    mag = (p.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) or p.get('aperio.AppMag')
           or p.get('mirax.GENERAL.OBJECTIVE_MAGNIFICATION') or 0)
    mpp = [p.get(k) for k in (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y)]
    wsi.close()
    mpp_x, mpp_y = (float(v) if v else None for v in mpp)
    return int(w), int(h), int(float(mag)), mpp_x, mpp_y


def read_ndpisplit(path, mag, x, y, ndpisplit):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ndpisplit, '-Ex%d,z0,%d,%d,%d,%d,tile' % (mag, x, y, TILE, TILE), '-O', tmp, '-cn', path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        out = glob.glob(os.path.join(tmp, '*.tif'))
        if len(out) != 1:
            raise RuntimeError('ndpisplit produced %d files for %s' % (len(out), ' '.join(cmd)))
        import tifffile
        return tifffile.imread(out[0])


_worker = {}


def _init_worker(path, backend, mag, ndpisplit):
    _worker.update(path=path, backend=backend, mag=mag, ndpisplit=ndpisplit)
    if backend == 'openslide':
        import openslide
        _worker['wsi'] = openslide.OpenSlide(path)


def read_tile(x, y):
    """256 x 256 RGB level-0 tile; pixels outside the scanned area are white."""
    if _worker['backend'] == 'openslide':
        rgba = np.array(_worker['wsi'].read_region((x, y), 0, (TILE, TILE)))
        rgb = rgba[..., :3].copy()
        rgb[rgba[..., 3] == 0] = 255
        return rgb
    return pad_white(read_ndpisplit(_worker['path'], _worker['mag'], x, y, _worker['ndpisplit']))


def write_tile(job):
    """Write one tile; returns True if the tile has enough tissue (written or present)."""
    out_path, x, y = job
    if os.path.exists(out_path):
        return True
    tile = read_tile(x, y)
    if tissue_fraction(tile) < MIN_TISSUE:
        return False
    tmp = out_path + '.part'
    Image.fromarray(tile).save(tmp, format='TIFF')
    os.replace(tmp, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--backend', choices=['ndpisplit', 'openslide'], default='openslide')
    ap.add_argument('--ndpisplit', default='ndpisplit', help='path to the ndpisplit executable')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    slides = sorted(f for f in os.listdir(args.wsi_dir) if f.lower().endswith(WSI_EXT))
    for f in slides:
        path = os.path.join(args.wsi_dir, f)
        slide = clean_name(os.path.splitext(f)[0])
        w, h, mag, mpp_x, mpp_y = slide_info(path, args.backend)
        if mag and mag != 40:
            print('%s: objective %dx, expected 40x' % (slide, mag))
        slide_dir = os.path.join(args.out_dir, slide)
        os.makedirs(slide_dir, exist_ok=True)
        jobs = [(os.path.join(slide_dir, tile_name(slide, x, y)), x, y) for x, y in tile_origins(w, h)]
        with Pool(args.workers, initializer=_init_worker, initargs=(path, args.backend, mag, args.ndpisplit)) as pool:
            kept = sum(pool.map(write_tile, jobs, chunksize=64))
        info = {'slide': slide, 'file': f, 'width': w, 'height': h, 'objective': mag,
                'mpp_x': mpp_x, 'mpp_y': mpp_y, 'tile': TILE, 'tiles_grid': len(jobs), 'tiles_kept': kept}
        with open(os.path.join(slide_dir, 'slide.json'), 'w') as fh:
            json.dump(info, fh, indent=1)
        print('%s: %d x %d, %d of %d tiles with >= %d%% tissue' % (slide, w, h, kept, len(jobs), MIN_TISSUE * 100))


if __name__ == '__main__':
    main()
