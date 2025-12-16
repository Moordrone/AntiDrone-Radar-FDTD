import yaml
from antidrone.radar.fmcw import simulate_iq_from_targets
from antidrone.radar.pipeline import RadarPipeline

def main():
    cfg = yaml.safe_load(open("config/default.yaml", "r", encoding="utf-8"))
    iq = simulate_iq_from_targets(
        fc_hz=cfg["radar"]["fc_hz"],
        bw_hz=cfg["radar"]["bw_hz"],
        t_chirp_s=cfg["radar"]["t_chirp_s"],
        fs_hz=cfg["radar"]["fs_hz"],
        n_chirps=cfg["radar"]["n_chirps"],
        targets=[
            {"r_m": 120.0, "v_mps": -8.0, "rcs": 1.0},
            {"r_m": 200.0, "v_mps": +3.0, "rcs": 0.6},
        ],
        snr_db=15.0
    )
    rad = RadarPipeline(cfg)
    dets, tracks = rad.step(iq)
    print("Detections (first 5):", dets[:5])
    print("Tracks:", tracks)

if __name__ == "__main__":
    main()
