from dataclasses import dataclass

@dataclass
class Grid3D:
    nx: int; ny: int; nz: int
    dx: float; dy: float; dz: float
    dt: float
