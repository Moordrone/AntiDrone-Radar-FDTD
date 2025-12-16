import numpy as np
from .cfar import cfar_2d_ca

def extract_detections(rd_mag, ranges_m, velocities_mps, guard=(2,2), train=(8,8), p_fa=1e-4, snr_floor_db=6.0):
    det_mask = cfar_2d_ca(rd_mag, guard=guard, train=train, p_fa=p_fa)

    noise = np.median(rd_mag)
    noise_db = 20*np.log10(noise + 1e-12)

    dets = []
    for idop, ir in zip(*np.where(det_mask)):
        amp = float(rd_mag[idop, ir])
        snr_db = 20*np.log10(amp + 1e-12) - noise_db
        if snr_db < snr_floor_db:
            continue
        dets.append({
            "range_m": float(ranges_m[ir]),
            "velocity_mps": float(velocities_mps[idop]),
            "snr_db": float(snr_db),
            "amp": amp,
            "bin": (int(idop), int(ir)),
        })
    return dets
