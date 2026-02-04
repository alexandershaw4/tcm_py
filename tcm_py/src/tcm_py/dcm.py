import numpy as np
from typing import Dict, Any, Tuple
from scipy.optimize import root

from tcm_py.model.tc_hilge2 import tc_hilge2
from tcm_py.model.tc_hilge2 import spm_vec, spm_unvec


# --------------------------------------------------
# Defaults / dimensions
# --------------------------------------------------

NPOP = 8     # SS, SP, SI, DP, DI, TP, RT, RC
NSTATES = 7 # V + 6 gating variables (AMPA, NMDA, GABA-A, GABA-B, M, H)


# --------------------------------------------------
# Parameter initialiser (zero / neutral)
# --------------------------------------------------

def make_default_P(ns: int) -> Dict[str, Any]:
    """
    Build a neutral, zero-parameter TCM parameter dict.

    Everything is zeroed but shaped correctly so:
    - tc_hilge2 runs
    - Laplace TF can build A, D
    - VL can pack/unpack later
    """

    # Extrinsic connectivity (MATLAB: P.A{1}..P.A{5})
    A = [np.zeros((ns, ns), dtype=float) for _ in range(5)]
    AN = [np.zeros((ns, ns), dtype=float) for _ in range(5)]

    # Intrinsic (within-source) connectivity
    # Shape: (npop, npop, ns)
    H = np.zeros((NPOP, NPOP, ns), dtype=float)
    Hn = np.zeros((NPOP, NPOP, ns), dtype=float)

    P = {
        # Connectivity
        "A": A,
        "AN": AN,
        "H": H,
        "Hn": Hn,

        # Input / observation gain
        "C": np.zeros(ns, dtype=float),

        # Time constants / rate parameters
        # MATLAB expects at least 4 columns; last two for M/H channels
        "T": np.zeros((NPOP, 6), dtype=float),

        # Membrane capacitance / gain per population
        "CV": np.ones(NPOP, dtype=float),

        # Excitability / bias
        "S": np.zeros(NPOP, dtype=float),

        # Background drive
        "E": np.zeros(NPOP, dtype=float),

        # Delays (scalars)
        "CT": 0.0,   # cortico-thalamic
        "TC": 0.0,   # thalamo-cortical
        "ID": 0.0,   # intrinsic delay

        # Noise / curvature parameters (Laplace TF expects this)
        # d[0]: delay penalty, d[1]: curvature, d[2]: smoothing
        "d": np.array([np.log(1e-6), np.log(1e-6), np.log(1.0)], dtype=float),

        # Misc flags (safe defaults)
        "scale_NMDA": 1.0,
        "endo": 1.0,
        "global": 0.0,
    }

    return P


# --------------------------------------------------
# State-space template
# --------------------------------------------------

def make_default_x(ns: int) -> np.ndarray:
    """
    Build a neutral state template: (ns, 8 pops, 7 states)
    Small non-zero voltage avoids flat sigmoid issues.
    """
    x = np.zeros((ns, NPOP, NSTATES), dtype=float)

    # Small depolarisation baseline for voltages
    x[..., 0] = -60.0  # mV-ish baseline, if your model uses that convention

    return x


# --------------------------------------------------
# Fixed point solver
# --------------------------------------------------

def find_fixed_point(
    P: Dict[str, Any],
    M: Dict[str, Any],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """
    Solve f(x) = 0 for the TCM using scipy.root on vectorised state space.
    """

    def fun(x_vec):
        x_struct = spm_unvec(x_vec, M["x"])
        f_vec, _, _ = tc_hilge2(
            x_struct,
            0.0,
            P,
            M,
            return_jacobian=False,
            return_delay=False,
        )
        return f_vec

    x_vec0 = spm_vec(x0)

    sol = root(fun, x_vec0, method="hybr", tol=tol, options={"maxfev": max_iter})

    if not sol.success:
        if verbose:
            print("⚠️ Fixed point solver did not fully converge:")
            print(sol.message)

    x_fp = spm_unvec(sol.x, x0)

    if verbose:
        res_norm = np.linalg.norm(fun(sol.x))
        print(f"Fixed point residual norm: {res_norm:.3e}")

    return x_fp


# --------------------------------------------------
# DCM-style model builder
# --------------------------------------------------

def build_dcm(
    ns: int,
    freqs: np.ndarray,
    find_fp: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    High-level DCM builder:
      - builds P
      - builds M
      - optionally finds fixed point

    Returns
    -------
    P, M
    """

    P = make_default_P(ns)
    x0 = make_default_x(ns)

    M = {
        "Hz": freqs,
        "x": x0.copy(),
        "f": tc_hilge2,
        "endogenous": True,
        "check_jacobian": False,
    }

    if find_fp:
        print("Solving for fixed point...")
        x_fp = find_fixed_point(P, M, x0)
        M["x"] = x_fp

    return P, M
