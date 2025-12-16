import numpy as np
from .fmcw import C0, fmcw_params

def simulate_iq_bistatic_from_rcs_response(
    fc_hz, bw_hz, t_chirp_s, fs_hz, n_chirps,
    tx_pos_m, rx_pos_m,
    tgt_pos0_m, tgt_vel_mps,
    resp_freq_hz, resp_sigma_m2, resp_phase_rad=None,
    snr_db=20.0, seed=0
):
    rng = np.random.default_rng(seed)
    p = fmcw_params(fc_hz, bw_hz, t_chirp_s)
    slope = p["slope"]
    lam   = p["lambda"]

    tx = np.array(tx_pos_m, float)
    rx = np.array(rx_pos_m, float)
    r0 = np.array(tgt_pos0_m, float)
    v  = np.array(tgt_vel_mps, float)

    n_samp = int(round(t_chirp_s * fs_hz))
    t_fast = np.arange(n_samp) / fs_hz
    t_slow = np.arange(n_chirps) * t_chirp_s

    f_inst = fc_hz + slope * t_fast
    sigma_t = np.interp(f_inst, resp_freq_hz, resp_sigma_m2, left=resp_sigma_m2[0], right=resp_sigma_m2[-1])
    amp_t = np.sqrt(np.maximum(sigma_t, 0.0))

    if resp_phase_rad is None:
        phi_t = 0.0
    else:
        phi_t = np.interp(f_inst, resp_freq_hz, resp_phase_rad, left=resp_phase_rad[0], right=resp_phase_rad[-1])

    iq = np.zeros((n_chirps, n_samp), dtype=np.complex128)

    for k, ts in enumerate(t_slow):
        pos = r0 + v * ts

        Rtx = np.linalg.norm(pos - tx)
        Rrx = np.linalg.norm(rx - pos)

        tau = (Rtx + Rrx) / C0
        f_b = slope * tau

        u_tx = (tx - pos) / (Rtx + 1e-15)
        u_rx = (rx - pos) / (Rrx + 1e-15)
        f_d = (1.0 / lam) * (v @ (u_tx + u_rx))

        att = 1.0 / ((Rtx * Rrx) + 1e-12)
        phase = 2*np.pi*(f_b * t_fast + f_d * ts) + phi_t
        iq[k, :] = (amp_t * att) * np.exp(1j * phase)

    sig_pow = np.mean(np.abs(iq)**2) + 1e-12
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (rng.normal(size=iq.shape) + 1j*rng.normal(size=iq.shape)) * np.sqrt(noise_pow/2)
    return iq + noise
