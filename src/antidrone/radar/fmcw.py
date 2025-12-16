import numpy as np

C0 = 299_792_458.0

def fmcw_params(fc_hz: float, bw_hz: float, t_chirp_s: float):
    slope = bw_hz / t_chirp_s
    lam = C0 / fc_hz
    return {"slope": slope, "lambda": lam}

def simulate_iq_from_targets(
    fc_hz: float,
    bw_hz: float,
    t_chirp_s: float,
    fs_hz: float,
    n_chirps: int,
    targets,
    snr_db: float = 20.0,
    seed: int = 0
):
    """
    targets: list of dicts: {"r_m":..., "v_mps":..., "rcs":...}
    Retour: iq [n_chirps, n_samples]
    Modèle beat FMCW baseband simple.
    """
    rng = np.random.default_rng(seed)
    p = fmcw_params(fc_hz, bw_hz, t_chirp_s)
    slope = p["slope"]
    lam = p["lambda"]

    n_samp = int(round(t_chirp_s * fs_hz))
    t_fast = np.arange(n_samp) / fs_hz
    t_slow = np.arange(n_chirps) * t_chirp_s

    iq = np.zeros((n_chirps, n_samp), dtype=np.complex128)

    for k, ts in enumerate(t_slow):
        s = np.zeros(n_samp, dtype=np.complex128)
        for tgt in targets:
            r0 = float(tgt["r_m"])
            v = float(tgt["v_mps"])
            a = float(tgt.get("rcs", 1.0))

            r = r0 + v * ts
            tau = 2.0 * r / C0

            f_b = slope * tau
            f_d = 2.0 * v / lam

            phase = 2.0 * np.pi * (f_b * t_fast + f_d * ts)
            s += a * np.exp(1j * phase)

        iq[k, :] = s

    sig_pow = np.mean(np.abs(iq) ** 2) + 1e-12
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = sig_pow / snr_lin
    noise = (rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape)) * np.sqrt(noise_pow / 2)
    return iq + noise
