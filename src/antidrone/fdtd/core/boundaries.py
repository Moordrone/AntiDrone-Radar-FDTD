import numpy as np

MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12

def _profile_sigma_kappa_alpha(n: int, npml: int, dx: float, dt: float,
                               m: int = 3, kappa_max: float = 6.0,
                               alpha_max: float = 0.05, R0: float = 1e-8):
    sigma = np.zeros(n, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)
    if npml <= 0:
        return sigma, kappa, alpha

    c0 = 299_792_458.0
    sigma_max = -(m + 1) * np.log(R0) * EPS0 * c0 / (2.0 * npml * dx)

    for i in range(npml):
        x = (npml - i) / npml
        s = sigma_max * (x ** m)
        k = 1.0 + (kappa_max - 1.0) * (x ** m)
        a = alpha_max * (1.0 - x)

        sigma[i] = s; kappa[i] = k; alpha[i] = a
        sigma[n - 1 - i] = s; kappa[n - 1 - i] = k; alpha[n - 1 - i] = a

    return sigma, kappa, alpha

def _bc_arrays(sigma, kappa, alpha, dt):
    b = np.exp(-((sigma / kappa) + alpha) * dt)
    denom = (sigma * kappa + (kappa ** 2) * alpha)
    denom = np.where(denom == 0.0, 1.0, denom)
    c = (sigma / denom) * (b - 1.0)
    return b, c

def _cross(a, b):
    return np.stack([
        a[...,1]*b[...,2] - a[...,2]*b[...,1],
        a[...,2]*b[...,0] - a[...,0]*b[...,2],
        a[...,0]*b[...,1] - a[...,1]*b[...,0],
    ], axis=-1)

class CPML3D:
    def __init__(self, nx, ny, nz, dx, dy, dz, dt, npml=10,
                 m=3, kappa_max=6.0, alpha_max=0.05, R0=1e-8):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.dt = dt
        self.npml = int(npml)

        sigx, kapx, alpx = _profile_sigma_kappa_alpha(nx, self.npml, dx, dt, m, kappa_max, alpha_max, R0)
        sigy, kapy, alpy = _profile_sigma_kappa_alpha(ny, self.npml, dy, dt, m, kappa_max, alpha_max, R0)
        sigz, kapz, alpz = _profile_sigma_kappa_alpha(nz, self.npml, dz, dt, m, kappa_max, alpha_max, R0)

        self.kx = kapx; self.ky = kapy; self.kz = kapz
        self.bx, self.cx = _bc_arrays(sigx, kapx, alpx, dt)
        self.by, self.cy = _bc_arrays(sigy, kapy, alpy, dt)
        self.bz, self.cz = _bc_arrays(sigz, kapz, alpz, dt)

        shp = (nx, ny, nz)
        self.psi_dEydz = np.zeros(shp); self.psi_dEzdy = np.zeros(shp); self.psi_dEzdx = np.zeros(shp)
        self.psi_dExdz = np.zeros(shp); self.psi_dExdy = np.zeros(shp); self.psi_dEydx = np.zeros(shp)
        self.psi_dHydz = np.zeros(shp); self.psi_dHzdy = np.zeros(shp); self.psi_dHzdx = np.zeros(shp)
        self.psi_dHxdz = np.zeros(shp); self.psi_dHxdy = np.zeros(shp); self.psi_dHydx = np.zeros(shp)

    def _apply_cpml_deriv(self, dF, psi, axis):
        if axis == 0:
            b = self.bx[:, None, None]; c = self.cx[:, None, None]; k = self.kx[:, None, None]
        elif axis == 1:
            b = self.by[None, :, None]; c = self.cy[None, :, None]; k = self.ky[None, :, None]
        else:
            b = self.bz[None, None, :]; c = self.cz[None, None, :]; k = self.kz[None, None, :]

        psi[:] = b * psi + c * dF
        return (dF / k) + psi

    def curlE_terms(self, dEydz, dEzdy, dEzdx, dExdz, dExdy, dEydx):
        dEydz = self._apply_cpml_deriv(dEydz, self.psi_dEydz, axis=2)
        dEzdy = self._apply_cpml_deriv(dEzdy, self.psi_dEzdy, axis=1)
        dEzdx = self._apply_cpml_deriv(dEzdx, self.psi_dEzdx, axis=0)
        dExdz = self._apply_cpml_deriv(dExdz, self.psi_dExdz, axis=2)
        dExdy = self._apply_cpml_deriv(dExdy, self.psi_dExdy, axis=1)
        dEydx = self._apply_cpml_deriv(dEydx, self.psi_dEydx, axis=0)
        return dEydz, dEzdy, dEzdx, dExdz, dExdy, dEydx

    def curlH_terms(self, dHydz, dHzdy, dHzdx, dHxdz, dHxdy, dHydx):
        dHydz = self._apply_cpml_deriv(dHydz, self.psi_dHydz, axis=2)
        dHzdy = self._apply_cpml_deriv(dHzdy, self.psi_dHzdy, axis=1)
        dHzdx = self._apply_cpml_deriv(dHzdx, self.psi_dHzdx, axis=0)
        dHxdz = self._apply_cpml_deriv(dHxdz, self.psi_dHxdz, axis=2)
        dHxdy = self._apply_cpml_deriv(dHxdy, self.psi_dHxdy, axis=1)
        dHydx = self._apply_cpml_deriv(dHydx, self.psi_dHydx, axis=0)
        return dHydz, dHzdy, dHzdx, dHxdz, dHxdy, dHydx
