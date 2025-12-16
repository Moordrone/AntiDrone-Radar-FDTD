import yaml
import numpy as np
from antidrone.radar.fmcw_bistatic_from_rcs import simulate_iq_bistatic_from_rcs_response
from antidrone.radar.pipeline import RadarPipeline

def test_bistatic_radar_smoke():
    cfg = yaml.safe_load(open("config/default.yaml", "r", encoding="utf-8"))
    f = np.linspace(23e9, 25e9, 512)
    sigma = 1e-2 * np.ones_like(f)
    phase = np.zeros_like(f)

    iq = simulate_iq_bistatic_from_rcs_response(
        fc_hz=cfg["radar"]["fc_hz"],
        bw_hz=cfg["radar"]["bw_hz"],
        t_chirp_s=cfg["radar"]["t_chirp_s"],
        fs_hz=cfg["radar"]["fs_hz"],
        n_chirps=cfg["radar"]["n_chirps"],
        tx_pos_m=(0.0, 0.0, 0.0),
        rx_pos_m=(50.0, 30.0, 0.0),
        tgt_pos0_m=(120.0, 10.0, 0.0),
        tgt_vel_mps=(-8.0, 0.0, 0.0),
        resp_freq_hz=f,
        resp_sigma_m2=sigma,
        resp_phase_rad=phase,
        snr_db=15.0,
        seed=0,
    )
    rad = RadarPipeline(cfg)
    dets, tracks = rad.step(iq)
    assert isinstance(dets, list)
    assert isinstance(tracks, list)
