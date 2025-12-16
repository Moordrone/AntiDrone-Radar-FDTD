from antidrone.fdtd.core.grid import Grid3D
from antidrone.fdtd.core.fields import Fields3D
from antidrone.fdtd.core.boundaries import CPML3D
from antidrone.fdtd.core.sources import PointSourceEz
import numpy as np

def test_fdtd_smoke_energy_grows():
    g = Grid3D(nx=30, ny=30, nz=30, dx=0.01, dy=0.01, dz=0.01, dt=1e-11)
    cpml = CPML3D(g.nx,g.ny,g.nz,g.dx,g.dy,g.dz,g.dt,npml=6)
    f = Fields3D(g, cpml=cpml)
    src = PointSourceEz(15,15,15, f0_hz=1e9, amp=1.0)
    e0 = float(np.mean(f.Ez**2))
    for n in range(30):
        f.update_h()
        f.update_e()
        src.inject(f, n)
    e1 = float(np.mean(f.Ez**2))
    assert e1 > e0
