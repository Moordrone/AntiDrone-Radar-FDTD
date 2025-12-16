import numpy as np
from antidrone.radar.fmcw import simulate_iq_from_targets
from antidrone.radar.range_doppler import range_doppler_map
from antidrone.radar.cfar import cfar_2d_ca

def test_radar_pipeline_smoke():
    iq = simulate_iq_from_targets(
        fc_hz=24e9, bw_hz=200e6, t_chirp_s=1e-3, fs_hz=2e6, n_chirps=64,
        targets=[{"r_m": 80.0, "v_mps": -5.0, "rcs": 1.0}],
        snr_db=15.0
    )
    mag, ranges, vels, _ = range_doppler_map(iq, 24e9, 200e6, 1e-3, 2e6, window="hann")
    det = cfar_2d_ca(mag, guard=(2,2), train=(6,6), p_fa=1e-3)
    assert det.shape == mag.shape
    assert np.any(det)
