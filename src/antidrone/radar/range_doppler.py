import numpy as np
from .fmcw import C0, fmcw_params

def _window(name: str, n: int):
    name = (name or "").lower()
    if name in ("hann", "hanning"):
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    return np.ones(n)

def range_doppler_map(iq: np.ndarray, fc_hz: float, bw_hz: float, t_chirp_s: float, fs_hz: float, window: str = "hann"):
    """
    iq: [n_chirps, n_samples]
    Retour:
      - RD magnitude [n_doppler, n_range]
      - ranges_m [n_range]
      - velocities_mps [n_doppler]
      - Xrd (complex)
    """
    n_ch, n_sa = iq.shape
    p = fmcw_params(fc_hz, bw_hz, t_chirp_s)

    w_r = _window(window, n_sa)
    w_d = _window(window, n_ch)

    Xr = np.fft.rfft(iq * w_r[None, :], axis=1)
    Xrd = np.fft.fftshift(np.fft.fft(Xr * w_d[:, None], axis=0), axes=0)

    mag = np.abs(Xrd)

    f_b = np.fft.rfftfreq(n_sa, d=1/fs_hz)
    ranges_m = C0 * f_b / (2 * p["slope"])

    f_d = np.fft.fftshift(np.fft.fftfreq(n_ch, d=t_chirp_s))
    velocities_mps = f_d * p["lambda"] / 2

    return mag, ranges_m, velocities_mps, Xrd
