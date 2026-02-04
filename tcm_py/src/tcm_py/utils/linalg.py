import numpy as np
from scipy import linalg

def solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stable linear solve."""
    return linalg.solve(A, B, assume_a='gen')

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5*(M + M.T)

def safe_cholesky(A: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Cholesky with diagonal jitter if needed."""
    A = np.array(A, dtype=float, copy=True)
    for k in range(10):
        try:
            return linalg.cholesky(A, lower=True, check_finite=False)
        except linalg.LinAlgError:
            A.flat[::A.shape[0]+1] += jitter*(10**k)
    raise

def denan(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    x = np.array(x, copy=True)
    x[~np.isfinite(x)] = fill
    return x
