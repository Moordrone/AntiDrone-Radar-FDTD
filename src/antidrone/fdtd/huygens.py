import numpy as np

MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12
C0 = 299_792_458.0
ETA0 = np.sqrt(MU0 / EPS0)

def _cross(a, b):
    return np.stack([
        a[...,1]*b[...,2] - a[...,2]*b[...,1],
        a[...,2]*b[...,0] - a[...,0]*b[...,2],
        a[...,0]*b[...,1] - a[...,1]*b[...,0],
    ], axis=-1)

class HuygensSurfaceRecorder:
    def __init__(self, x0, x1, y0, y1, z0, z1, decim: int = 1):
        self.x0, self.x1 = int(x0), int(x1)
        self.y0, self.y1 = int(y0), int(y1)
        self.z0, self.z1 = int(z0), int(z1)
        self.decim = max(1, int(decim))
        self._k = 0
        self.frames = []

    def sample(self, fields):
        if (self._k % self.decim) != 0:
            self._k += 1
            return
        self._k += 1

        Ex,Ey,Ez = fields.Ex, fields.Ey, fields.Ez
        Hx,Hy,Hz = fields.Hx, fields.Hy, fields.Hz
        x0,x1,y0,y1,z0,z1 = self.x0,self.x1,self.y0,self.y1,self.z0,self.z1

        fr = {
            "x0": (Ex[x0, y0:y1, z0:z1], Ey[x0, y0:y1, z0:z1], Ez[x0, y0:y1, z0:z1],
                   Hx[x0, y0:y1, z0:z1], Hy[x0, y0:y1, z0:z1], Hz[x0, y0:y1, z0:z1]),
            "x1": (Ex[x1-1, y0:y1, z0:z1], Ey[x1-1, y0:y1, z0:z1], Ez[x1-1, y0:y1, z0:z1],
                   Hx[x1-1, y0:y1, z0:z1], Hy[x1-1, y0:y1, z0:z1], Hz[x1-1, y0:y1, z0:z1]),

            "y0": (Ex[x0:x1, y0, z0:z1], Ey[x0:x1, y0, z0:z1], Ez[x0:x1, y0, z0:z1],
                   Hx[x0:x1, y0, z0:z1], Hy[x0:x1, y0, z0:z1], Hz[x0:x1, y0, z0:z1]),
            "y1": (Ex[x0:x1, y1-1, z0:z1], Ey[x0:x1, y1-1, z0:z1], Ez[x0:x1, y1-1, z0:z1],
                   Hx[x0:x1, y1-1, z0:z1], Hy[x0:x1, y1-1, z0:z1], Hz[x0:x1, y1-1, z0:z1]),

            "z0": (Ex[x0:x1, y0:y1, z0], Ey[x0:x1, y0:y1, z0], Ez[x0:x1, y0:y1, z0],
                   Hx[x0:x1, y0:y1, z0], Hy[x0:x1, y0:y1, z0], Hz[x0:x1, y0:y1, z0]),
            "z1": (Ex[x0:x1, y0:y1, z1-1], Ey[x0:x1, y0:y1, z1-1], Ez[x0:x1, y0:y1, z1-1],
                   Hx[x0:x1, y0:y1, z1-1], Hy[x0:x1, y0:y1, z1-1], Hz[x0:x1, y0:y1, z1-1]),
        }
        self.frames.append(fr)

    def to_frequency(self, dt):
        n = len(self.frames)
        if n < 8:
            raise ValueError("Pas assez d'échantillons surface pour une FFT utile.")
        faces = ["x0","x1","y0","y1","z0","z1"]
        spec = {}
        for face in faces:
            Ets = []
            Hts = []
            for fr in self.frames:
                ex,ey,ez,hx,hy,hz = fr[face]
                Ets.append(np.stack([ex,ey,ez], axis=-1))
                Hts.append(np.stack([hx,hy,hz], axis=-1))
            Ets = np.stack(Ets, axis=0)
            Hts = np.stack(Hts, axis=0)
            Ef = np.fft.rfft(Ets, axis=0)
            Hf = np.fft.rfft(Hts, axis=0)
            spec[face] = (Ef, Hf)
        freq = np.fft.rfftfreq(n, d=dt*self.decim)
        return freq, spec

def far_field_from_huygens(spec, freq_hz, g, box, rhat):
    x0,x1,y0,y1,z0,z1 = box
    dx,dy,dz = g.dx,g.dy,g.dz

    rhat = np.array(rhat, dtype=np.float64)
    rhat = rhat / (np.linalg.norm(rhat) + 1e-15)

    Nf = len(freq_hz)
    Efar = np.zeros((Nf,3), dtype=np.complex128)

    faces = [
        ("x0", np.array([-1,0,0]), "yz"),
        ("x1", np.array([+1,0,0]), "yz"),
        ("y0", np.array([0,-1,0]), "xz"),
        ("y1", np.array([0,+1,0]), "xz"),
        ("z0", np.array([0,0,-1]), "xy"),
        ("z1", np.array([0,0,+1]), "xy"),
    ]

    for face, nvec, plane in faces:
        Ef, Hf = spec[face]

        if plane == "yz":
            ix = x0 if face == "x0" else (x1-1)
            ys = (np.arange(y0,y1) + 0.5) * dy
            zs = (np.arange(z0,z1) + 0.5) * dz
            Y,Z = np.meshgrid(ys, zs, indexing="ij")
            X = np.full_like(Y, (ix+0.5)*dx)
            dS = dy*dz
        elif plane == "xz":
            iy = y0 if face == "y0" else (y1-1)
            xs = (np.arange(x0,x1) + 0.5) * dx
            zs = (np.arange(z0,z1) + 0.5) * dz
            X,Z = np.meshgrid(xs, zs, indexing="ij")
            Y = np.full_like(X, (iy+0.5)*dy)
            dS = dx*dz
        else:
            iz = z0 if face == "z0" else (z1-1)
            xs = (np.arange(x0,x1) + 0.5) * dx
            ys = (np.arange(y0,y1) + 0.5) * dy
            X,Y = np.meshgrid(xs, ys, indexing="ij")
            Z = np.full_like(X, (iz+0.5)*dz)
            dS = dx*dy

        rdot = rhat[0]*X + rhat[1]*Y + rhat[2]*Z
        n = nvec.astype(np.float64); n = n / (np.linalg.norm(n)+1e-15)

        Js = _cross(n, Hf)
        Ms = -_cross(n, Ef)

        term1 = _cross(rhat, Ms)
        term2 = _cross(rhat, _cross(rhat, Js))
        integrand = term1 + ETA0 * term2

        for k, f in enumerate(freq_hz):
            if f <= 0:
                continue
            w = 2*np.pi*f
            k0 = w / C0
            ph = np.exp(1j * k0 * rdot)
            s = np.sum(integrand[k] * ph[...,None]) * dS
            Efar[k] += 1j * k0 * (ETA0 / (4*np.pi)) * s

    return Efar

def rcs_from_far(Efar, Ei_spec, pol_hat):
    pol_hat = np.array(pol_hat, dtype=np.float64)
    pol_hat = pol_hat / (np.linalg.norm(pol_hat) + 1e-15)
    Es = Efar @ pol_hat
    num = np.abs(Es)**2
    den = np.abs(Ei_spec)**2 + 1e-30
    return 4*np.pi * (num / den), np.angle(Es)
