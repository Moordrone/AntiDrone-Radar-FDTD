from .range_doppler import range_doppler_map
from .detect import extract_detections
from .tracker_manager import TrackerManager

class RadarPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = TrackerManager(
            dt=cfg["radar"]["t_chirp_s"],
            gate_m=cfg.get("tracking", {}).get("gate_m", 5.0)
        )

    def step(self, iq):
        mag, ranges, vels, _ = range_doppler_map(
            iq,
            fc_hz=self.cfg["radar"]["fc_hz"],
            bw_hz=self.cfg["radar"]["bw_hz"],
            t_chirp_s=self.cfg["radar"]["t_chirp_s"],
            fs_hz=self.cfg["radar"]["fs_hz"],
            window=self.cfg.get("processing", {}).get("window", "hann"),
        )

        cfar_cfg = self.cfg.get("processing", {}).get("cfar", {})
        dets = extract_detections(
            mag, ranges, vels,
            guard=tuple(cfar_cfg.get("guard", (2,2))),
            train=tuple(cfar_cfg.get("train", (8,8))),
            p_fa=float(cfar_cfg.get("p_fa", 1e-4)),
            snr_floor_db=float(cfar_cfg.get("snr_floor_db", 6.0))
        )

        tracks = self.tracker.update(dets)
        return dets, tracks
