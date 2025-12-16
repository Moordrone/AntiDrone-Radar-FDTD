import numpy as np
from .grid import Grid3D

MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12

class Fields3D:
    def __init__(self, g: Grid3D, cpml=None):
        self.g = g
        self.cpml = cpml
        nx, ny, nz = g.nx, g.ny, g.nz

        self.Ex = np.zeros((nx, ny, nz), dtype=np.float64)
        self.Ey = np.zeros((nx, ny, nz), dtype=np.float64)
        self.Ez = np.zeros((nx, ny, nz), dtype=np.float64)
        self.Hx = np.zeros((nx, ny, nz), dtype=np.float64)
        self.Hy = np.zeros((nx, ny, nz), dtype=np.float64)
        self.Hz = np.zeros((nx, ny, nz), dtype=np.float64)

        self.eps_r = np.ones((nx, ny, nz), dtype=np.float64)
        self.mu_r  = np.ones((nx, ny, nz), dtype=np.float64)
        self.sigma = np.zeros((nx, ny, nz), dtype=np.float64)

    def update_h(self):
        g = self.g
        dt = g.dt
        dx, dy, dz = g.dx, g.dy, g.dz
        mu = MU0 * self.mu_r

        dEydz = (np.roll(self.Ey, -1, axis=2) - self.Ey) / dz
        dEzdy = (np.roll(self.Ez, -1, axis=1) - self.Ez) / dy
        dEzdx = (np.roll(self.Ez, -1, axis=0) - self.Ez) / dx
        dExdz = (np.roll(self.Ex, -1, axis=2) - self.Ex) / dz
        dExdy = (np.roll(self.Ex, -1, axis=1) - self.Ex) / dy
        dEydx = (np.roll(self.Ey, -1, axis=0) - self.Ey) / dx

        if self.cpml is not None:
            dEydz, dEzdy, dEzdx, dExdz, dExdy, dEydx = self.cpml.curlE_terms(
                dEydz, dEzdy, dEzdx, dExdz, dExdy, dEydx
            )

        self.Hx -= (dt / mu) * (dEydz - dEzdy)
        self.Hy -= (dt / mu) * (dEzdx - dExdz)
        self.Hz -= (dt / mu) * (dExdy - dEydx)

    def update_e(self):
        g = self.g
        dt = g.dt
        dx, dy, dz = g.dx, g.dy, g.dz
        eps = EPS0 * self.eps_r

        dHydz = (self.Hy - np.roll(self.Hy, 1, axis=2)) / dz
        dHzdy = (self.Hz - np.roll(self.Hz, 1, axis=1)) / dy
        dHzdx = (self.Hz - np.roll(self.Hz, 1, axis=0)) / dx
        dHxdz = (self.Hx - np.roll(self.Hx, 1, axis=2)) / dz
        dHxdy = (self.Hx - np.roll(self.Hx, 1, axis=1)) / dy
        dHydx = (self.Hy - np.roll(self.Hy, 1, axis=0)) / dx

        if self.cpml is not None:
            dHydz, dHzdy, dHzdx, dHxdz, dHxdy, dHydx = self.cpml.curlH_terms(
                dHydz, dHzdy, dHzdx, dHxdz, dHxdy, dHydx
            )

        ca = (1.0 - (self.sigma * dt) / (2.0 * eps)) / (1.0 + (self.sigma * dt) / (2.0 * eps))
        cb = (dt / eps) / (1.0 + (self.sigma * dt) / (2.0 * eps))

        self.Ex = ca * self.Ex + cb * (dHydz - dHzdy)
        self.Ey = ca * self.Ey + cb * (dHzdx - dHxdz)
        self.Ez = ca * self.Ez + cb * (dHxdy - dHydx)
