import numpy as np
from .fmcw import C0, fmcw_params

def simulate_iq_from_rcs_response(
    fc_hz: float, bw_hz: float, t_chirp_s: float, fs_hz: float, n_chirps: int,
    r_m: float, v_mps: float,
    resp_freq_hz: np.ndarray, resp_sigma_m2: np.ndarray, resp_phase_rad: np.ndarray | None = None,
    snr_db: float = 20.0, seed: int = 0
):
    rng = np.random.default_rng(seed)
    p = fmcw_params(fc_hz, bw_hz, t_chirp_s)
    slope = p["slope"]
    lam = p["lambda"]

    n_samp = int(round(t_chirp_s * fs_hz))
    t_fast = np.arange(n_samp) / fs_hz
    t_slow = np.arange(n_chirps) * t_chirp_s

    f_inst = fc_hz + slope * t_fast
    sigma_t = np.interp(f_inst, resp_freq_hz, resp_sigma_m2, left=resp_sigma_m2[0], right=resp_sigma_m2[-1])
    amp_t = np.sqrt(np.maximum(sigma_t, 0.0)) / (r_m**2 + 1e-12)

    if resp_phase_rad is None:
        phi_t = 0.0
    else:
        phi_t = np.interp(f_inst, resp_freq_hz, resp_phase_rad, left=resp_phase_rad[0], right=resp_phase_rad[-1])

    iq = np.zeros((n_chirps, n_samp), dtype=np.complex128)

    for k, ts in enumerate(t_slow):
        r = r_m + v_mps * ts
        tau = 2.0 * r / C0
        f_b = slope * tau
        f_d = 2.0 * v_mps / lam
        phase = 2.0 * np.pi * (f_b * t_fast + f_d * ts) + phi_t
        iq[k, :] = amp_t * np.exp(1j * phase)

    sig_pow = np.mean(np.abs(iq)**2) + 1e-12
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (rng.normal(size=iq.shape) + 1j*rng.normal(size=iq.shape)) * np.sqrt(noise_pow/2)
    return iq + noise
