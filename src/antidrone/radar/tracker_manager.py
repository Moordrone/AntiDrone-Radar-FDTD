from .tracking import Kalman1D

class TrackerManager:
    def __init__(self, dt, gate_m=5.0):
        self.dt = float(dt)
        self.gate = float(gate_m)
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        used = set()

        # association nearest-neighbor en range (baseline)
        for tid, trk in list(self.tracks.items()):
            trk.predict()
            best = None
            best_dist = self.gate
            for i, det in enumerate(detections):
                if i in used:
                    continue
                d = abs(det["range_m"] - trk.r)
                if d < best_dist:
                    best = i
                    best_dist = d
            if best is not None:
                trk.update(detections[best]["range_m"])
                used.add(best)

        # création nouvelles pistes
        for i, det in enumerate(detections):
            if i in used:
                continue
            kf = Kalman1D(self.dt)
            kf.init(det["range_m"], det["velocity_mps"])
            self.tracks[self.next_id] = kf
            self.next_id += 1

        return self.get_tracks()

    def get_tracks(self):
        return [{"track_id": tid, "range_m": trk.r, "velocity_mps": trk.v}
                for tid, trk in self.tracks.items()]
