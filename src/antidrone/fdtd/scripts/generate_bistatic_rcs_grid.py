import json
import numpy as np
import argparse
import yaml

from antidrone.fdtd.postprocess import compute_rcs_bistatic
from antidrone.fdtd.huygens import HuygensSurfaceRecorder
from antidrone.fdtd.core.grid import Grid3D
from antidrone.fdtd.core.fields import Fields3D
from antidrone.fdtd.core.boundaries import CPML3D
from antidrone.fdtd.core.sources import TFSFPlaneWaveX
from antidrone.fdtd.geometry import drone_box_like

def sph_dir(theta_deg: float, phi_deg: float):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], dtype=float)

def run_fdtd_and_record_surface(g, npml, pad, f0_hz, inc_amp, n_steps, huygens_box, with_target: bool):
    cpml = CPML3D(g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.dt, npml=npml)
    f = Fields3D(g, cpml=cpml)

    if with_target:
        drone_box_like(f, center=(int(0.55*g.nx), g.ny//2, g.nz//2), size=(14,10,8), eps_r=3.0, sigma=0.0)

    x0 = npml + pad; x1 = g.nx - (npml + pad)
    y0 = npml + pad; y1 = g.ny - (npml + pad)
    z0 = npml + pad; z1 = g.nz - (npml + pad)
    src = TFSFPlaneWaveX(x0,x1,y0,y1,z0,z1, f0_hz=f0_hz, amp=inc_amp)

    hx0,hx1,hy0,hy1,hz0,hz1 = huygens_box
    hs = HuygensSurfaceRecorder(hx0,hx1,hy0,hy1,hz0,hz1, decim=1)

    for n in range(n_steps):
        f.update_h(); src.apply(f, n)
        f.update_e(); src.apply(f, n)
        hs.sample(f)

    freq_hz, spec = hs.to_frequency(dt=g.dt)
    return freq_hz, spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="rcs_database/bistatic_demo.npz")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    fd = cfg["fdtd"]

    if args.quick:
        grid_kwargs = dict(nx=70, ny=56, nz=56, dx=fd["dx"], dy=fd["dy"], dz=fd["dz"], dt=fd["dt"])
        n_steps = 600
        theta = np.linspace(30, 150, 9)
        phi   = np.linspace(0, 315, 8)
    else:
        grid_kwargs = dict(nx=fd["nx"], ny=fd["ny"], nz=fd["nz"], dx=fd["dx"], dy=fd["dy"], dz=fd["dz"], dt=fd["dt"])
        n_steps = int(fd["n_steps"])
        theta = np.linspace(10, 170, 17)
        phi   = np.linspace(0, 330, 12)

    g = Grid3D(**grid_kwargs)
    npml = int(fd.get("npml", 10))
    pad = int(fd.get("pad", 6))
    f0_hz = float(fd.get("f0_hz", 2e9))
    inc_amp = float(fd.get("inc_amp", 1.0))

    # Huygens box around target (auto)
    cx, cy, cz = int(0.55*g.nx), g.ny//2, g.nz//2
    huygens_box = (max(8,cx-18), min(g.nx-8,cx+18),
                   max(8,cy-16), min(g.ny-8,cy+16),
                   max(8,cz-16), min(g.nz-8,cz+16))

    freq, spec_with = run_fdtd_and_record_surface(g, npml, pad, f0_hz, inc_amp, n_steps, huygens_box, with_target=True)
    _, spec_ref = run_fdtd_and_record_surface(g, npml, pad, f0_hz, inc_amp, n_steps, huygens_box, with_target=False)

    spec_diff = {}
    for face in spec_with.keys():
        Ef_w, Hf_w = spec_with[face]
        Ef_r, Hf_r = spec_ref[face]
        spec_diff[face] = (Ef_w - Ef_r, Hf_w - Hf_r)

    Nf = len(freq)
    sigma = np.zeros((Nf, len(theta), len(phi)), dtype=np.float64)
    phase = np.zeros_like(sigma)

    for it, th in enumerate(theta):
        for ip, ph in enumerate(phi):
            rx_dir = sph_dir(th, ph)
            resp = compute_rcs_bistatic(freq, spec_diff, g, huygens_box,
                                       incident_dir=(+1,0,0), rx_dir=rx_dir, pol_hat=(0,0,1))
            sigma[:, it, ip] = resp.sigma_m2
            phase[:, it, ip] = resp.phase_rad

    meta = {
        "grid": grid_kwargs,
        "npml": npml,
        "pad": pad,
        "f0_hz": f0_hz,
        "inc_amp": inc_amp,
        "n_steps": n_steps,
        "incident_dir": [1,0,0],
        "pol_hat": [0,0,1],
        "huygens_box": list(huygens_box),
        "note": "Baseline bistatic dataset from Huygens NTF; spec_diff = with - ref."
    }

    np.savez_compressed(
        args.out,
        freq_hz=freq,
        theta_deg=np.array(theta, dtype=np.float64),
        phi_deg=np.array(phi, dtype=np.float64),
        sigma_m2=sigma,
        phase_rad=phase,
        meta_json=json.dumps(meta)
    )
    print("[OK] saved:", args.out, "sigma shape:", sigma.shape)

if __name__ == "__main__":
    main()
