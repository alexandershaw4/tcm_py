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

def make_voltage_only_J(ns: int, NPOP: int, NSTATES: int, Vw: np.ndarray):
    """
    Build observation weight vector P['J'] of length ns*NPOP*NSTATES,
    with weights only on voltage states.
    """
    assert Vw.size == NPOP, "Voltage weight vector must be length NPOP"

    x_tmp = np.zeros((ns, NPOP, NSTATES), dtype=float)

    # voltage state is index 0
    for p in range(NPOP):
        x_tmp[0, p, 0] = Vw[p]

    # vectorise exactly as the model does
    Jvec = spm_vec(x_tmp)

    return Jvec

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
    Gb = np.zeros((NPOP, NPOP, ns), dtype=float)

    Vw = np.array([
        0.2000,
        0.9900,
        0.1000,
        0.8000,
        0.1000,
        0.2000,
        0.0500,
        0.1000
    ], dtype=float)

    P = {
        # Connectivity
        "A": A,
        "AN": AN,
        "H": H,
        "Hn": Hn,
        "Gb": Gb,

        # Input / observation gain
        "C": np.zeros(ns, dtype=float),

        # Time constants / rate parameters
        # MATLAB expects at least 4 columns; last two for M/H channels
        "T": np.zeros((NPOP, 6), dtype=float),

        # Membrane capacitance / gain per population
        "CV": np.zeros(NPOP, dtype=float),

        # Excitability / bias
        "S": np.zeros(NPOP, dtype=float),

        # Background drive
        "E": np.zeros(NPOP, dtype=float),

        # Delays (scalars)
        "CT": 0.0,   # cortico-thalamic
        "TC": 0.0,   # thalamo-cortical
        "ID": 0.0,   # intrinsic delay

        "L": 1.0,  # gain

        # Noise / curvature parameters (Laplace TF expects this)
        # d[0]: delay penalty, d[1]: curvature, d[2]: smoothing
        "d": np.array([np.log(1e-6), np.log(1e-6), np.log(1.0)], dtype=float),

        # Misc flags (safe defaults)
        "scale_NMDA": 1.0,
        "endo": 1.0,
        "global": 0.0,
    }

    # ---- set observation weights (THIS is the important bit)
    P["J"] = make_voltage_only_J(
        ns=ns,
        NPOP=NPOP,
        NSTATES=NSTATES,
        Vw=Vw
    )

    return P


# --------------------------------------------------
# State-space template
# --------------------------------------------------

def make_default_x(ns: int) -> np.ndarray:
    x = np.zeros((ns, 8, 7), dtype=float)

    x[..., 0] = np.array([-52.0007, -53.7096, -52.5124, -53.2265,
                          -52.9797, -52.0864, -51.3178, -51.9504])

    x[..., 1] = np.array([4.2965, 3.5260, 2.1694, 2.1643,
                          2.3391, 2.3391, 2.9868, 4.2624])

    x[..., 2] = np.array([5.3863, 5.2125, 2.6731, 2.7433,
                          2.9519, 3.8767, 3.5779, 6.3515])

    x[..., 3] = np.array([4.3279, 4.6363, 3.2798, 2.1643,
                          2.3391, 2.3391, 2.9868, 4.2624])

    x[..., 4] = np.array([5.3863, 5.2125, 2.6731, 2.7433,
                          2.9519, 3.8767, 3.5779, 6.3515])

    x[..., 5] = np.array([1.3566, 0.5643, 1.0693, 0.7391,
                          0.8434, 1.3058, 1.7889, 1.3868])

    x[..., 6] = np.array([0.0, 0.0, 0.0, 0.0,
                          0.0, 4.0, 0.0, 4.0])

    return x


#def make_default_x(ns: int) -> np.ndarray:
    """
    Build a neutral state template: (ns, 8 pops, 7 states)
    Small non-zero voltage avoids flat sigmoid issues.
    """
#    x = np.zeros((ns, NPOP, NSTATES), dtype=float)

    # Small depolarisation baseline for voltages
    x[..., 0] = -70.0  # mV-ish baseline, if your model uses that convention

 #   return x


# --------------------------------------------------
# Fixed point solver
# --------------------------------------------------

def find_fixed_point(
    P: Dict[str, Any],
    M: Dict[str, Any],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """
    Robust fixed-point solver for tc_hilge2 using damped Newton + line-search on ||f||.
    Falls back to scipy.root(krylov) if needed.

    Assumes:
      - spm_vec(x_struct) -> flat (n,)
      - spm_unvec(x_vec, M["x"]) -> structured state like M["x"]
      - tc_hilge2(x_struct, u, P, M, return_jacobian=True, return_delay=False)
        returns f_vec (flat), J (2D), _
    """

    # ---- helpers
    def vec(x_struct):
        return spm_vec(x_struct).reshape(-1)

    def unvec(x_vec):
        return spm_unvec(x_vec, M["x"])

    def fun_and_jac(x_vec):
        x_struct = unvec(x_vec)
        f_vec, J, _ = tc_hilge2(
            x_struct,
            0.0,   # if your model expects an input vector, change this
            P,
            M,
            return_jacobian=True,
            return_delay=False,
        )
        f_vec = np.asarray(f_vec, dtype=float).reshape(-1)
        J = np.asarray(J, dtype=float)
        return f_vec, J

    # ---- init
    x = vec(x0).copy()
    f, J = fun_and_jac(x)
    nf = np.linalg.norm(f)

    # variable scaling to reduce conditioning issues:
    # scale_i ~ 1 / (|x_i| + 1) is a decent generic choice
    scale = 1.0 / (np.abs(x) + 1.0)
    # prevent extreme scaling
    scale = np.clip(scale, 1e-3, 1e3)

    if verbose:
        print(f"[fp] init ||f|| = {nf:.3e}")

    # ---- damped Newton loop
    # We minimise phi(x) = 0.5 ||f(x)||^2, using Newton step from J.
    for it in range(max_iter):
        if not np.isfinite(nf):
            if verbose:
                print("[fp] non-finite ||f||; aborting to fallback.")
            break
        if nf < tol:
            if verbose:
                print(f"[fp] converged at iter {it}, ||f||={nf:.3e}")
            return unvec(x)

        # Solve (J_scaled) dx_scaled = -f, then unscale dx
        # This improves robustness when states have different magnitudes.
        # Let x = S^{-1} z (or z = S x). Here we implement a simple right-scaling:
        # J (dx) = -f, with dx = (dxs / scale)
        Js = J / scale[None, :]   # columns scaled
        fs = f.copy()

        # Ridge for stability (esp. near-singular J)
        lam = 1e-8 + 1e-6 * (nf / (np.linalg.norm(x) + 1e-12))
        A = Js.T @ Js + lam * np.eye(Js.shape[1])
        b = -Js.T @ fs
        dxs = np.linalg.solve(A, b)     # step in scaled coordinates
        dx = dxs / scale               # unscale

        # Backtracking line search on ||f|| (Armijo-like on norm)
        t = 1.0
        nf0 = nf
        accept = False
        for ls in range(25):
            x_new = x + t * dx
            f_new, J_new = fun_and_jac(x_new)
            nf_new = np.linalg.norm(f_new)

            if np.isfinite(nf_new) and (nf_new <= nf0 * (1.0 - 1e-4 * t) or nf_new < nf0):
                x, f, J, nf = x_new, f_new, J_new, nf_new
                accept = True
                break
            t *= 0.5

        if verbose:
            msg = "ok" if accept else "fail"
            print(f"[fp] it {it:03d} ||f||={nf0:.3e} -> {nf:.3e}  step={t:.2e}  {msg}")

        if not accept:
            # If Newton can't find a decreasing step, bail to fallback solver
            break

        # small-step termination (can help if nf plateaus)
        if np.linalg.norm(t * dx) < 1e-10 * (np.linalg.norm(x) + 1.0):
            if verbose:
                print("[fp] step too small; bailing to fallback.")
            break

    # ---- fallback: Krylov root (more forgiving for large systems)
    def fun_only(x_vec):
        f_vec, _J = fun_and_jac(x_vec)
        return f_vec

    sol = root(fun_only, x, method="krylov", options={"maxiter": 200, "fatol": tol})

    if verbose:
        print(f"[fp] fallback success={sol.success}  msg={sol.message}")

    x_fp = unvec(sol.x)
    res_norm = np.linalg.norm(fun_only(sol.x))

    if verbose:
        print(f"[fp] final residual ||f|| = {res_norm:.3e}")

    # ---- safety checks (tune thresholds to your model scale)
    bad = (
        (not np.isfinite(res_norm)) or
        (res_norm > max(1e-2, 10 * tol)) or
        (not np.all(np.isfinite(vec(x_fp)))) or
        (np.linalg.norm(vec(x_fp)) > 1e8)
    )
    if bad:
        if verbose:
            print("[fp] rejecting fixed point; returning x0.")
        return x0

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
