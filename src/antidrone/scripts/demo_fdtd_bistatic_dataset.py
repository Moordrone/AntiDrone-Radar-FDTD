import argparse, json
import numpy as np
import yaml

from antidrone.fdtd.core.grid import Grid3D
from antidrone.fdtd.core.fields import Fields3D
from antidrone.fdtd.core.boundaries import CPML3D
from antidrone.fdtd.core.sources import TFSFPlaneWaveX
from antidrone.fdtd.geometry import drone_box_like
from antidrone.fdtd.huygens import HuygensSurfaceRecorder
from antidrone.fdtd.postprocess import compute_rcs_bistatic

def sph_dir(theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], dtype=float)

def run_once(cfg, quick=False):
    fd = cfg["fdtd"]
    if quick:
        # plus léger pour Colab
        nx, ny, nz = 70, 56, 56
        n_steps = 600
    else:
        nx, ny, nz = fd["nx"], fd["ny"], fd["nz"]
        n_steps = fd["n_steps"]

    g = Grid3D(nx=nx, ny=ny, nz=nz, dx=fd["dx"], dy=fd["dy"], dz=fd["dz"], dt=fd["dt"])
    npml = int(fd.get("npml", 10))
    pad = int(fd.get("pad", 6))

    cpml = CPML3D(g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.dt, npml=npml)
    f = Fields3D(g, cpml=cpml)

    # drone proxy
    drone_box_like(f, center=(int(0.55*nx), ny//2, nz//2), size=(12, 9, 7), eps_r=3.0, sigma=0.0)

    # TF/SF box
    x0 = npml + pad; x1 = g.nx - (npml + pad)
    y0 = npml + pad; y1 = g.ny - (npml + pad)
    z0 = npml + pad; z1 = g.nz - (npml + pad)
    src = TFSFPlaneWaveX(x0,x1,y0,y1,z0,z1, f0_hz=fd["f0_hz"], amp=fd["inc_amp"])

    # Huygens surface around target
    hs = HuygensSurfaceRecorder(
        x0=max(10, int(0.45*nx)), x1=min(nx-10, int(0.75*nx)),
        y0=max(10, ny//2-14), y1=min(ny-10, ny//2+14),
        z0=max(10, nz//2-14), z1=min(nz-10, nz//2+14),
        decim=1
    )

    # run WITH target
    for n in range(n_steps):
        f.update_h(); src.apply(f, n)
        f.update_e(); src.apply(f, n)
        hs.sample(f)

    freq, spec_with = hs.to_frequency(dt=g.dt)

    # run REF (no target)
    f2 = Fields3D(g, cpml=CPML3D(g.nx,g.ny,g.nz,g.dx,g.dy,g.dz,g.dt,npml=npml))
    src2 = TFSFPlaneWaveX(x0,x1,y0,y1,z0,z1, f0_hz=fd["f0_hz"], amp=fd["inc_amp"])
    hs2 = HuygensSurfaceRecorder(hs.x0,hs.x1,hs.y0,hs.y1,hs.z0,hs.z1, decim=1)

    for n in range(n_steps):
        f2.update_h(); src2.apply(f2, n)
        f2.update_e(); src2.apply(f2, n)
        hs2.sample(f2)

    _, spec_ref = hs2.to_frequency(dt=g.dt)

    # spec_diff
    spec_diff = {}
    for face in spec_with.keys():
        Ef_w, Hf_w = spec_with[face]
        Ef_r, Hf_r = spec_ref[face]
        spec_diff[face] = (Ef_w - Ef_r, Hf_w - Hf_r)

    # compute for a couple of bistatic directions
    for th, ph in [(90,0), (90,90), (60,45)]:
        rx_dir = sph_dir(th, ph)
        resp = compute_rcs_bistatic(freq, spec_diff, g, (hs.x0,hs.x1,hs.y0,hs.y1,hs.z0,hs.z1),
                                    incident_dir=(+1,0,0), rx_dir=rx_dir, pol_hat=(0,0,1))
        print(f"bistatic (theta={th},phi={ph}) sigma[1:5] =", resp.sigma_m2[1:5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--quick", action="store_true", help="run a lighter demo for Colab")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_once(cfg, quick=args.quick)

if __name__ == "__main__":
    main()
