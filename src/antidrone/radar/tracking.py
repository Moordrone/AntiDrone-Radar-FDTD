import numpy as np

class Kalman1D:
    """Track range + range_rate (v) : x=[r, v]."""
    def __init__(self, dt, sigma_r=1.0, sigma_a=1.0):
        self.dt = float(dt)
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        q = float(sigma_a)**2
        dt = self.dt
        self.Q = q * np.array([[dt**4/4, dt**3/2],
                               [dt**3/2, dt**2]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[float(sigma_r)**2]])
        self.x = np.zeros((2,1))
        self.P = np.eye(2) * 1e3
        self.initialized = False

    def init(self, r0, v0=0.0):
        self.x[:] = [[float(r0)], [float(v0)]]
        self.P = np.eye(2)
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, r_meas):
        z = np.array([[float(r_meas)]])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def r(self): return float(self.x[0,0])
    @property
    def v(self): return float(self.x[1,0])
