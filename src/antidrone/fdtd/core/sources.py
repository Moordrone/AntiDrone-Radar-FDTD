import numpy as np

C0  = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12

def ricker(t, f0):
    a = (np.pi * f0 * (t - 1.0 / f0))
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)

class PointSourceEz:
    def __init__(self, ix, iy, iz, f0_hz, t0_s=0.0, amp=1.0):
        self.ix, self.iy, self.iz = int(ix), int(iy), int(iz)
        self.f0 = float(f0_hz)
        self.t0 = float(t0_s)
        self.amp = float(amp)

    def inject(self, fields, n_step):
        t = n_step * fields.g.dt - self.t0
        fields.Ez[self.ix, self.iy, self.iz] += self.amp * ricker(t, self.f0)

class TFSFPlaneWaveX:
    """
    TF/SF onde plane propageant +x, polarisation Ez, champ Hy associé.
    Boîte [x0:x1, y0:y1, z0:z1] (bornes Python: x0 inclus, x1 exclu).
    """
    def __init__(self, x0, x1, y0, y1, z0, z1, f0_hz, amp=1.0, t0_s=0.0):
        self.x0, self.x1 = int(x0), int(x1)
        self.y0, self.y1 = int(y0), int(y1)
        self.z0, self.z1 = int(z0), int(z1)
        self.f0 = float(f0_hz)
        self.amp = float(amp)
        self.t0 = float(t0_s)
        self.eta0 = np.sqrt(MU0 / EPS0)

    def _Ez_inc(self, t):
        return self.amp * ricker(t - self.t0, self.f0)

    def _Hy_inc(self, t):
        return self._Ez_inc(t) / self.eta0

    def apply(self, fields, n_step):
        g = fields.g
        dt = g.dt
        dx = g.dx

        t_e = n_step * dt
        t_h = (n_step + 0.5) * dt

        def delay_for_x(ix):
            return (ix - self.x0) * dx / C0

        # H corrections (Hy) at x0 and x1
        ix = self.x0
        hy_inc = self._Hy_inc(t_h - delay_for_x(ix))
        fields.Hy[ix-1, self.y0:self.y1, self.z0:self.z1] -= hy_inc
        fields.Hy[ix,   self.y0:self.y1, self.z0:self.z1] += hy_inc

        ix = self.x1
        hy_inc = self._Hy_inc(t_h - delay_for_x(ix))
        fields.Hy[ix-1, self.y0:self.y1, self.z0:self.z1] += hy_inc
        fields.Hy[ix,   self.y0:self.y1, self.z0:self.z1] -= hy_inc

        # E corrections (Ez) at x0 and x1
        ix = self.x0
        ez_inc = self._Ez_inc(t_e - delay_for_x(ix))
        fields.Ez[ix, self.y0:self.y1, self.z0:self.z1] += ez_inc

        ix = self.x1
        ez_inc = self._Ez_inc(t_e - delay_for_x(ix))
        fields.Ez[ix-1, self.y0:self.y1, self.z0:self.z1] -= ez_inc
