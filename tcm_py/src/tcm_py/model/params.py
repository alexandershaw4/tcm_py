from dataclasses import dataclass
import numpy as np

@dataclass
class Priors:
    m0: np.ndarray
    S0: np.ndarray
