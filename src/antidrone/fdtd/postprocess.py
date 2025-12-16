import numpy as np
from .scattering_response import ScatteringResponse
from .huygens import far_field_from_huygens, rcs_from_far

def incident_spectrum_proxy(freq_hz, f0_hz, amp=1.0):
    f0 = float(f0_hz)
    return amp * np.exp(-((freq_hz - f0)/(0.6*f0 + 1e-12))**2)

def compute_rcs_monostatic(freq_hz, spec, grid, box, incident_dir=(+1,0,0), pol_hat=(0,0,1), f0_hz=2e9, inc_amp=1.0):
    inc_dir = np.array(incident_dir, dtype=float)
    inc_dir /= (np.linalg.norm(inc_dir) + 1e-15)
    rhat = -inc_dir  # backscatter

    Efar = far_field_from_huygens(spec, freq_hz, grid, box, rhat=rhat)
    Ei = incident_spectrum_proxy(freq_hz, f0_hz=f0_hz, amp=inc_amp)
    sigma, phase = rcs_from_far(Efar, Ei_spec=Ei, pol_hat=pol_hat)

    return ScatteringResponse(freq_hz=freq_hz, sigma_m2=sigma.astype(np.float64), phase_rad=phase.astype(np.float64),
                             meta={"type":"monostatic_ntf", "incident_dir":incident_dir, "pol_hat":pol_hat})

def compute_rcs_bistatic(freq_hz, spec, grid, box, incident_dir=(+1,0,0), rx_dir=(0,1,0), pol_hat=(0,0,1), Ei_spec=None):
    rhat = np.array(rx_dir, dtype=float)
    rhat /= (np.linalg.norm(rhat) + 1e-15)

    Efar = far_field_from_huygens(spec, freq_hz, grid, box, rhat=rhat)
    if Ei_spec is None:
        Ei_spec = np.ones_like(freq_hz, dtype=np.complex128)

    sigma, phase = rcs_from_far(Efar, Ei_spec=Ei_spec, pol_hat=pol_hat)
    return ScatteringResponse(freq_hz=freq_hz, sigma_m2=sigma.astype(np.float64), phase_rad=phase.astype(np.float64),
                             meta={"type":"bistatic_ntf", "incident_dir":incident_dir, "rx_dir":rx_dir, "pol_hat":pol_hat})
