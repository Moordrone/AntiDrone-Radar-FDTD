from dataclasses import dataclass
import numpy as np

@dataclass
class ScatteringResponse:
    freq_hz: np.ndarray
    sigma_m2: np.ndarray
    phase_rad: np.ndarray | None = None
    meta: dict | None = None
