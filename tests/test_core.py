import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pasc_consol import analysis_table, composite, descriptors, masks  # noqa: E402
from pasc_consol.constants import (COMPOSITE_INPUTS, COVARIATE_TRAINING_CONSTANTS,  # noqa: E402
                                   LABEL_ADC, LABEL_SCC, TRAINING_CONSTANTS)


MPP = (10.0, 10.0)  # mapping scale of a 0.25 um/px scan (x 40)


def square(lab, r, c, size, value):
    lab[r:r + size, c:c + size] = value


def test_descriptors_two_regions():
    lab = np.full((200, 200), LABEL_ADC, np.uint8)
    square(lab, 10, 10, 60, LABEL_SCC)    # 3600 px
    square(lab, 120, 120, 30, LABEL_SCC)  # 900 px
    square(lab, 190, 5, 2, LABEL_SCC)     # 4 px island: not counted, stays in the denominator
    d = descriptors.slide_descriptors(lab, MPP)
    total = 3600 + 900 + 4
    assert d['A_SCC_px'] == total and d['T_px'] == 200 * 200 and d['N_px'] == 200 * 200
    assert d['CCCount'] == 2 and d['LargestRegion_px'] == 3600
    assert d['LesionCount'] == 2 and d['Multifocal'] == 1
    assert d['DCR'] == pytest.approx(3600 / total)
    assert d['SecondClusterFrac'] == pytest.approx(900 / total)
    assert d['eligible'] and d['T_um2'] == pytest.approx(4e6)  # 40,000 px x 100 um2


def test_small_regions_count_for_second_fraction_only():
    lab = np.zeros((100, 100), np.uint8)
    square(lab, 0, 0, 20, LABEL_SCC)      # 400 px
    square(lab, 50, 50, 3, LABEL_SCC)     # 9 px: below 10 px, but 2.2% second region
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['CCCount'] == 1
    assert d['SecondClusterFrac'] == pytest.approx(9 / 409)
    assert d['LesionCount'] == 1 and d['Multifocal'] == 0


def test_eight_connectivity_joins_diagonals():
    lab = np.zeros((50, 50), np.uint8)
    for i in range(20):
        lab[5 + i, 5 + i] = LABEL_SCC
    assert descriptors.region_sizes(lab == LABEL_SCC).tolist() == [20]


def test_significance_threshold():
    lab = np.full((300, 300), LABEL_ADC, np.uint8)
    square(lab, 0, 0, 100, LABEL_SCC)     # 10000 px
    square(lab, 200, 200, 20, LABEL_SCC)  # 400 px = 3.8% of SCC -> retained, not significant
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['CCCount'] == 2 and d['LesionCount'] == 1 and d['Multifocal'] == 0


def test_eligibility_threshold():
    lab = np.zeros((1000, 1000), np.uint8)
    square(lab, 0, 0, 109, LABEL_SCC)     # 11881 px x 100 um2 < 1.2e6 um2
    assert not descriptors.slide_descriptors(lab, MPP)['eligible']
    square(lab, 0, 0, 110, LABEL_SCC)     # 12100 px x 100 um2
    assert descriptors.slide_descriptors(lab, MPP)['eligible']
    assert descriptors.slide_descriptors(lab, (0.25, 0.25))['eligible'] is False  # level-0 pixels


@pytest.mark.parametrize('tumor_pixels, eligible', [(11999, False), (12000, True), (12001, True)])
@pytest.mark.parametrize('scc_pixels', [0, 3, 6000])
def test_exact_mapping_scale_eligibility(tumor_pixels, eligible, scc_pixels):
    lab = np.full((150, 150), 255, np.uint8)
    flat = lab.ravel()
    flat[:scc_pixels] = LABEL_SCC
    flat[scc_pixels:tumor_pixels] = LABEL_ADC
    flat[tumor_pixels:tumor_pixels + 100] = 0
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['T_px'] == tumor_pixels
    assert d['T_um2'] == tumor_pixels * 100
    assert d['eligible'] is eligible


def test_eligibility_uses_calibrated_pixel_area():
    lab = np.full((100, 120), LABEL_ADC, np.uint8)
    assert descriptors.slide_descriptors(lab, (5.0, 20.0))['eligible']
    assert not descriptors.slide_descriptors(lab, (5.0, 10.0))['eligible']


def test_ignore_is_not_sampled_tissue():
    lab = np.full((200, 200), 255, np.uint8)
    square(lab, 0, 0, 150, 0)             # 22500 px sampled tissue
    square(lab, 0, 0, 120, LABEL_ADC)     # 14400 px tumor
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['N_px'] == 22500 and d['T_px'] == 14400
    p = descriptors.patient_descriptors([d])
    assert p['V_raw'] == pytest.approx(np.log(14400 / (22500 - 14400)))


def test_slide_without_scc():
    lab = np.full((200, 200), LABEL_ADC, np.uint8)
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['eligible'] and d['DCR'] == 0 and d['CCCount'] == 0 and d['Multifocal'] == 0


def test_interface_and_front():
    lab = np.zeros((100, 100), np.uint8)
    square(lab, 20, 20, 60, LABEL_ADC)
    square(lab, 40, 40, 20, LABEL_SCC)    # SCC fully inside ADC
    d = descriptors.slide_descriptors(lab, MPP)
    assert d['InterfaceFrac'] == pytest.approx(1.0)
    assert d['FrontFrac'] == pytest.approx(1.0)
    assert 20 < d['FrontDistance_px'] < 31
    assert d['ShapeFactor'] > 1


def slide(**kw):
    base = {'eligible': True, 'N_px': 1000, 'T_px': 100, 'A_SCC_px': 50, 'DCR': 0.5,
            'SecondClusterFrac': 0.1, 'CCCount': 5, 'LesionCount': 2, 'Multifocal': 1,
            'ShapeFactor': np.e, 'InterfaceFracSym': 0.4, 'FrontFrac': 1.0, 'FrontDistance_px': np.e - 1}
    return {**base, **kw}


def test_patient_aggregation():
    a = slide(T_px=300, A_SCC_px=100, DCR=0.8, SecondClusterFrac=0.1, CCCount=5, LesionCount=2, Multifocal=1)
    b = slide(T_px=100, A_SCC_px=100, DCR=0.4, SecondClusterFrac=0.3, CCCount=9, LesionCount=1, Multifocal=0)
    c = slide(eligible=False, CCCount=1000)
    p = descriptors.patient_descriptors([a, b, c])
    assert p['n_eligible_slides'] == 2
    assert p['SCC_in_tumor_pt'] == pytest.approx(0.5)
    assert p['C_raw'] == pytest.approx(0.0)
    assert p['D1_raw'] == pytest.approx(1 - (0.8 * 300 + 0.4 * 100) / 400)
    assert p['D2_raw'] == 1
    assert p['D3_raw'] == pytest.approx(np.log(10))
    assert p['D4_raw'] == pytest.approx(np.log(3))
    assert p['D5_raw'] == pytest.approx(0.3)
    assert p['B_raw'] == pytest.approx(1.0)
    assert p['S3_raw'] == pytest.approx(-1.0)
    occ = (0.3 * 300 + 0.1 * 100) / 400
    assert p['V_raw'] == pytest.approx(np.log(occ / (1 - occ)))


def test_label_png_and_mpp(tmp_path):
    from PIL import Image
    lab = np.zeros((2, 2), np.uint8)
    lab[0, 0] = LABEL_SCC
    lab[1, 1] = 255
    Image.fromarray(lab, mode='L').save(tmp_path / 'S_labels.png')
    (tmp_path / 'S.json').write_text('{"mpp_x": 10.0, "mpp_y": 10.0}')
    back, mpp = masks.load_slide(tmp_path / 'S_labels.png', tmp_path / 'S.json')
    assert (back == lab).all() and mpp == (10.0, 10.0)
    assert masks.load_mpp(tmp_path / 'missing.json', default=(10.0, 10.0)) == (10.0, 10.0)
    Image.fromarray(np.full((4, 4), 7, np.uint8), mode='L').save(tmp_path / 'bad.png')
    with pytest.raises(ValueError):
        masks.load_label_png(tmp_path / 'bad.png')


def test_composite_at_training_means_is_zero():
    row = pd.DataFrame([{c: TRAINING_CONSTANTS[c][0] for c in COMPOSITE_INPUTS}])
    assert composite.mapped_consolidation(row).iloc[0] == pytest.approx(0.0)


def test_composite_direction():
    # more dispersion (higher D inputs) -> lower consolidation
    base = {c: TRAINING_CONSTANTS[c][0] for c in COMPOSITE_INPUTS}
    hi = dict(base, D1_raw=base['D1_raw'] + TRAINING_CONSTANTS['D1_raw'][1])
    z = composite.mapped_consolidation(pd.DataFrame([hi])).iloc[0]
    assert z == pytest.approx(-1 / 5 / 0.439442)


def test_visual_rounding():
    df = pd.DataFrame({'Human_consol_P1': [1, 2, 2, 3], 'Human_consol_P2': [1, 2, 3, 4],
                       'Human_consol_P3': [2, 3, 3, 4]})
    v = composite.visual_consolidation(df)
    assert v['Human_consol_mean'].tolist() == pytest.approx([4 / 3, 7 / 3, 8 / 3, 11 / 3])
    assert v['Visual_rounded'].tolist() == [1, 2, 3, 4]


def test_t_category():
    size = pd.Series([2.0, 2.1, 4.0, 4.1, 3.0])
    archived = pd.Series([1, 2, 3, 3, 4])
    size_t, final_t = analysis_table.t_category(size, archived)
    assert size_t.tolist() == [1, 2, 2, 3, 2]
    assert final_t.tolist() == [1, 2, 2, 3, 4]


def test_analysis_table_derive():
    df = pd.DataFrame({
        'Human_consol_P1': [2, 2], 'Human_consol_P2': [2, 3], 'Human_consol_P3': [2, 3],
        'SCC_in_tumor_pt': [0.5, 0.25], 'C_raw': [0.0, np.log(1 / 3)],
        **{c: [TRAINING_CONSTANTS[c][0]] * 2 for c in COMPOSITE_INPUTS},
        'tumor_size_cm_archived': [3.851515, 5.0], 'archived_pT_exact_for_audit': [2, 3],
        'pT_ge3_original': [0.0, np.nan], 'Age': [64.242424, 70], 'Eligible_slides': [6, 8],
        'SCC_proportion_PATHOLOGIST': [55, 40]})
    d = analysis_table.derive(df)
    assert d['Human_high'].tolist() == [0, 1]           # 2.0 is not above the training median
    assert d['Reader23_mean'].tolist() == [2.0, 3.0]
    assert d['Consolidation_z'].tolist() == pytest.approx([0.0, 0.0])
    assert d['Consol_high'].tolist() == [1, 1]
    assert d['pT_changed_or_filled'].tolist() == [0, 1]  # missing original pT counts as filled
    assert d['size_z'].iloc[0] == pytest.approx(0.0, abs=1e-6)
    mu, sd = COVARIATE_TRAINING_CONSTANTS['Human_consol_mean']
    assert d['visual_z'].iloc[1] == pytest.approx((8 / 3 - mu) / sd)
    assert d['path10'].tolist() == [5.5, 4.0]
