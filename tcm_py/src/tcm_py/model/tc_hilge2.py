"""Extended canonical thalamo-cortical conductance-based neural-mass model.

Python port of `tc_hilge2.m` (Dr Alexander Shaw, 2020).

The model has 8 populations per source:
1 SS (L4 spiny stellate), 2 SP (L2/3 superficial pyramids), 3 SI (L2/3 interneurons),
4 DP (L5 deep pyramids), 5 DI (L5 deep interneurons), 6 TP (L6 thal proj pyramids),
7 RT (reticular), 8 RC (relay).

States per population (nk) are expected to include at least:
1 V (voltage)
2 gE (AMPA)
3 gI (GABA-A)
4 gN (NMDA)
5 gB (GABA-B)
Optionally (if nk>=7):
6 gM (M-current)
7 gH (H-current)

This implementation:
* preserves SPM/MATLAB vectorisation order (column-major / Fortran order)
* provides a finite-difference Jacobian when requested (safe default)
* reproduces the MATLAB delay-matrix logic
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

def spm_unvec(v: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Inverse of spm_vec: reshape vector into template shape (Fortran order)."""
    v = np.asarray(v, dtype=float).reshape(-1, order="F")
    return v.reshape(np.asarray(template).shape, order="F")

def spm_vec(arr: np.ndarray) -> np.ndarray:
    """SPM-style vectorisation (Fortran order)."""
    return np.asarray(arr, order="F").reshape(-1, order="F")


def reshape_x(x: np.ndarray, x_template: np.ndarray) -> np.ndarray:
    """Reshape vectorised x into template shape using Fortran order."""
    x = np.asarray(x)
    if x.shape == x_template.shape:
        return x.astype(float, copy=False)
    return x.reshape(x_template.shape, order="F").astype(float, copy=False)


def as_3d(mat: np.ndarray, ns: int) -> np.ndarray:
    """Ensure intrinsic matrices are (npop,npop,ns)."""
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 3:
        return mat
    if mat.ndim == 2:
        return np.repeat(mat[:, :, None], ns, axis=2)
    raise ValueError(f"Expected 2D or 3D matrix, got shape {mat.shape}")


def rates_from_T(T: np.ndarray, col: int, ns: int, npop: int) -> np.ndarray:
    """Extract per-source/per-population rate array from P['T'].

    MATLAB code uses KE(i,:) etc; in practice P.T can be:
    - (ns, ncol) => one value per source, broadcast to pops
    - (npop, ncol) => one value per pop, broadcast to sources
    - (ns, npop, ncol) => fully specified
    """
    T = np.asarray(T)
    if T.ndim == 3:
        out = T[:, :, col]
        if out.shape != (ns, npop):
            raise ValueError(f"P.T[:,:,{col}] must be (ns,npop) but got {out.shape}")
        return out
    if T.ndim == 2:
        if T.shape[0] == ns:
            v = T[:, col].reshape(ns, 1)
            return np.repeat(v, npop, axis=1)
        if T.shape[0] == npop:
            v = T[:, col].reshape(1, npop)
            return np.repeat(v, ns, axis=0)
    if T.ndim == 1:
        val = float(T[col])
        return np.full((ns, npop), val)
    #if T.ndim == 1:
    #    # single value
    #    #return np.repeat(T[col].reshape(1, 1), ns, axis=0).repeat(npop, axis=1)
    #    if T.ndim == 1:
    #        val = float(T[col])
    #        return np.full((ns, npop), val)
    raise ValueError(f"Unsupported P.T shape {T.shape}")

def logistic(z):
    # clip to avoid exp overflow; 60 is plenty (exp(60) ~ 1e26)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def vector_param(p: Any, ns: int, npop: int) -> np.ndarray:
    """Coerce parameter to per-population vector."""
    a = np.asarray(p, dtype=float)
    if a.size == 1:
        return np.full((npop,), float(a))
    if a.shape == (npop,):
        return a
    if a.shape == (ns, npop):
        # caller should index per source
        return a
    raise ValueError(f"Cannot coerce parameter with shape {a.shape} to per-population")


def local_mag_block(v: np.ndarray, scale_nmda: float) -> np.ndarray:
    """Magnesium block used in the MATLAB code (IncludeMH branch).

    MATLAB uses a backslash (mldivide) on a diagonal-like denominator. For scalars,
    this is element-wise reciprocal. We implement the same computation.
    """
    denom = 1.0 + 0.2 * np.exp(-0.062 * scale_nmda * v)
    return 1.0 / denom

# the actuakl TCM equations of motion
def tc_hilge2(
    x: np.ndarray,
    u: Any,
    P: Dict[str, Any],
    M: Dict[str, Any],
    return_jacobian: bool = False,
    return_delay: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Compute state derivatives (and optionally Jacobian and delay matrix)."""

    # Allow P['p'] wrapping
    if isinstance(P, dict) and "p" in P and isinstance(P["p"], dict):
        P = P["p"]
    
    verbose=False

    # Optional M/H included in this file is set to 1 in the MATLAB code
    include_mh = True

    x_template = np.asarray(M["x"], dtype=float)
    ns, npop, nk = x_template.shape
    X = reshape_x(x, x_template)  # (ns,npop,nk)

    # --- Extrinsic connections and gains
    A = []
    AN = []
    for i in range(len(P.get("A", []))):
        A.append(np.exp(np.asarray(P["A"][i], dtype=float)))
        AN.append(np.exp(np.asarray(P["AN"][i], dtype=float)))
    C = np.exp(np.asarray(P.get("C", np.zeros((ns, 1))), dtype=float))
    if C.ndim == 1:
        C = C.reshape(-1, 1)

    # Damp reciprocal lateral strengths on A (as in MATLAB)
    for i in range(len(A)):
        Ai = A[i]
        L = (Ai > np.exp(-8)) & (Ai.T > np.exp(-8))
        A[i] = Ai / (1.0 + 8.0 * L)

    # --- Intrinsic strengths
    G = np.exp(as_3d(np.asarray(P.get("H"), dtype=float), ns))
    Gn = np.exp(as_3d(np.asarray(P.get("Hn"), dtype=float), ns))
    if "Gb" in P:
        Gb = np.exp(as_3d(np.asarray(P.get("Gb"), dtype=float), ns))
    else:
        Gb = G

    # --- Fixed extrinsic routing masks
    SA = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    ) / 8.0
    SA[:, [2, 3, 4]] = 0.0  # ketamine study restriction

    SNMDA = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    ) / 8.0
    SNMDA[:, [2, 3, 4]] = 0.0

    # --- Fixed intrinsic topology
    GEa = np.array(
        [
            [0, 0, 0, 0, 0, 2, 0, 2],
            [2, 2, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 2, 0, 0],
        ],
        dtype=float,
    )
    GEn = np.array(
        [
            [0, 0, 0, 0, 0, 2, 0, 2],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 2, 0, 0],
        ],
        dtype=float,
    )
    GIa = np.array(
        [
            [8, 0, 10, 0, 0, 0, 0, 0],
            [0, 18, 10, 0, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0, 0, 0],
            [0, 0, 0, 8, 6, 0, 0, 0],
            [0, 0, 0, 0, 14, 0, 0, 0],
            [0, 0, 0, 0, 6, 8, 0, 0],
            [0, 0, 0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 8, 8],
        ],
        dtype=float,
    )
    GIb = GIa.copy()

    # --- Channel decay rates
    T = np.asarray(P.get("T"), dtype=float)
    KE = np.exp(-rates_from_T(T, 0, ns, npop)) * 1000.0 / 2.2
    KI = np.exp(-rates_from_T(T, 1, ns, npop)) * 1000.0 / 5.0
    KN = np.exp(-rates_from_T(T, 2, ns, npop)) * 1000.0 / 100.0
    KB = np.exp(-rates_from_T(T, 3, ns, npop)) * 1000.0 / 300.0

    if "T1" in P:
        KE = KE + float(np.asarray(P["T1"])[0])
        KN = KN + float(np.asarray(P["T1"])[1])

    # Reversal potentials
    VL, VE, VI, VR0, VN, VB = -70.0, 60.0, -90.0, -52.0, 10.0, -100.0

    if include_mh:
        VM, VH = -52.0, -30.0
        Mh = np.asarray(P.get("Mh", np.zeros((npop,))), dtype=float).reshape(-1)
        Hh = np.asarray(P.get("Hh", np.zeros((npop,))), dtype=float).reshape(-1)
        GIm_diag = 4.0 * np.ones((npop,)) * np.exp(Mh)
        GIh_mask = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=float)
        GIh_diag = 4.0 * GIh_mask * np.exp(Hh)
        KM = np.exp(-rates_from_T(T, 4, ns, npop)) * 1000.0 / 160.0
        KH = np.exp(-rates_from_T(T, 5, ns, npop)) * 1000.0 / 100.0

    # Capacitances and leak
    CV_base = np.array([128 * 3, 128, 64, 128, 64, 128, 64, 128], dtype=float) / 1000.0
    CVp = np.exp(np.asarray(P.get("CV", np.zeros((npop,))), dtype=float)).reshape(-1)
    if CVp.size == 1:
        CVp = np.repeat(CVp, npop)
    CV = CVp * CV_base
    #GL = 1.0
    #GL = float(np.exp(np.asarray(P.get("leak", 0.0), dtype=float)))
    leak = float(np.asarray(P.get("leak", 0.0), dtype=float))
    GL = float(np.exp(np.clip(leak, -5.0, 5.0)))

    # Mean-field excitability shifts & firing
    #VR = VR0 + np.exp(np.asarray(P.get("S", 0.0), dtype=float))
    #S = float(np.asarray(P.get("S", 0.0), dtype=float))
    #VR = VR0 + np.exp(np.clip(S, -5.0, 5.0))

    S_raw = np.asarray(P.get("S", 0.0), dtype=float)

    # S can be scalar, (npop,), (ns, npop), etc.
    if S_raw.ndim == 0:
        S_eff = S_raw  # scalar
    elif S_raw.size == 1:
        S_eff = float(S_raw.reshape(-1)[0])
    else:
        # Try to interpret as per-population vector
        S_eff = S_raw.reshape(-1)

    # Build VR with broadcasting
    if np.ndim(S_eff) == 0:
        VR = VR0 + np.exp(np.clip(S_eff, -5.0, 5.0))
    else:
        # per-population; make shape (1, npop) then broadcast with V (ns, npop)
        S_eff = np.clip(S_eff, -5.0, 5.0)
        VR = VR0 + np.exp(S_eff)[None, :]

    R = 2.0 / 3.0
    V = X[:, :, 0]
    V = np.clip(V, -120.0, 60.0)
    #FF = 1.0 / (1.0 + np.exp(-R * (V - VR)))
    #FF = logistic(R * (V - VR))
    RS = 30.0
    #FF = np.where(V >= VR, 1.0, FF)
    #FF = np.where(V >= RS, 0.0, FF)
    #FF = np.where(V >= VR, 1.0, FF)
    #FF = np.where(V >= RS, 0.0, FF)
    k_on  = 2.0/3.0        # your R
    k_off = 1.0            # tune; higher = sharper cutoff
    FF_on  = logistic(k_on  * (V - VR))
    FF_off = logistic(k_off * (V - RS))   # 0 below RS, ->1 above RS
    FF = FF_on * (1.0 - FF_off)

    m = FF

    if include_mh:
        #h = 1.0 - 1.0 / (1.0 + np.exp(-(2.0 / 3.0) * (V - VH)))
        h  = 1.0 - logistic((2.0 / 3.0) * (V - VH))

    # Extrinsic effects per source
    a = np.zeros((ns, 5), dtype=float)
    an = np.zeros((ns, 5), dtype=float)
    if len(A) >= 5:
        a[:, 0] = (A[0] @ m[:, 1])
        a[:, 1] = (A[1] @ m[:, 3])
        a[:, 2] = (A[2] @ m[:, 5])
        a[:, 3] = (A[3] @ m[:, 6])
        a[:, 4] = (A[4] @ m[:, 7])
        an[:, 0] = (AN[0] @ m[:, 1])
        an[:, 1] = (AN[1] @ m[:, 3])
        an[:, 2] = (AN[2] @ m[:, 5])
        an[:, 3] = (AN[3] @ m[:, 6])
        an[:, 4] = (AN[4] @ m[:, 7])

    BE = np.exp(np.asarray(P.get("E", 0.0), dtype=float)) * 0.8

    # Optional global scaling of intrinsic blocks
    if "global" in P:
        g = np.asarray(P["global"], dtype=float).reshape(-1)
        if g.size >= 4:
            GEa = GEa * np.exp(g[0])
            GEn = GEn * np.exp(g[1])
            GIa = GIa * np.exp(g[2])
            GIb = GIb * np.exp(g[3])

    # Output derivative structure
    F = np.zeros_like(X)

    # Coerce u
    u_arr = np.asarray(u, dtype=float).reshape(-1)

    scale_nmda = float(np.exp(np.asarray(P.get("scale_NMDA", 0.0), dtype=float)))

    for i in range(ns):
        dU = u_arr * float(C[i, 0])

        Gi = G[:, :, i]
        Gni = Gn[:, :, i]
        Gbi = Gb[:, :, i]

        WiE = Gi * GEa
        WiN = Gni * GEn
        WiI = Gi * GIa
        WiB = Gbi * GIb

        mi = m[i, :]
        E = WiE @ mi
        EN = WiN @ mi
        I = WiI @ mi
        IB = WiB @ mi

        if include_mh and nk >= 7:
            Im = GIm_diag * mi
            Ih = GIh_diag * h[i, :]

        # Extrinsic + background (x2)
        E = (E + BE + SA @ a[i, :]) * 2.0
        EN = (EN + BE + SNMDA @ an[i, :]) * 2.0

        if "endo" in P:
            endo = np.asarray(P["endo"], dtype=float).reshape(-1)
            if endo.size >= 1:
                E[1] = E[1] + 2.0 * np.exp(endo[0])

        # Exogenous input routing
        if u_arr.size > 1:
            E[7] = E[7] + dU[1]
            E[1] = E[1] + dU[0]
        else:
            # thalamus relay & reticular
            E[[7, 6]] = E[[7, 6]] + dU[0]

        if "thi" in P:
            thi = float(np.exp(np.asarray(P["thi"], dtype=float)))
            E[7] = E[7] + thi
            EN[7] = EN[7] + thi

        # --- Voltage equation
        Vi = X[i, :, 0]
        Vi = np.clip(Vi, -120.0, 60.0)
        gE = X[i, :, 1]
        gI = X[i, :, 2]
        gN = X[i, :, 3]
        gB = X[i, :, 4]

        if include_mh and nk >= 7:
            gM = X[i, :, 5]
            gH = X[i, :, 6]
            mag = local_mag_block(Vi, scale_nmda)
            F[i, :, 0] = (
                GL * (VL - Vi)
                + gE * (VE - Vi)
                + gI * (VI - Vi)
                + gB * (VB - Vi)
                + gM * (VM - Vi)
                + gH * (VH - Vi)
                + gN * (VN - Vi) * mag
            ) / CV
        else:
            # If nk<7, drop M/H terms
            mag = local_mag_block(Vi, scale_nmda)
            F[i, :, 0] = (
                GL * (VL - Vi)
                + gE * (VE - Vi)
                + gI * (VI - Vi)
                + gB * (VB - Vi)
                + gN * (VN - Vi) * mag
            ) / CV

        # --- Conductance dynamics
        F[i, :, 1] = (E - gE) * KE[i, :]
        F[i, :, 2] = (I - gI) * KI[i, :]
        F[i, :, 4] = (IB - gB) * KB[i, :]
        F[i, :, 3] = (EN - gN) * KN[i, :]

        if include_mh and nk >= 7:
            F[i, :, 5] = (Im - gM) * KM[i, :]
            F[i, :, 6] = (Ih - gH) * KH[i, :]
        
        if verbose:
            rV = np.linalg.norm(F[:, :, 0])
            rE = np.linalg.norm(F[:, :, 1])
            rI = np.linalg.norm(F[:, :, 2])
            rN = np.linalg.norm(F[:, :, 3])
            rB = np.linalg.norm(F[:, :, 4])
            print("res blocks | V,E,I,N,B:", rV, rE, rI, rN, rB)

    f_vec = spm_vec(F)

    J = None
    D = None

    if return_jacobian:
        # Finite-difference Jacobian in vectorised space (safe; analytic can be added later)
        eps = float(P.get("fd_eps", 1e-6))
        n = f_vec.size
        J = np.zeros((n, n), dtype=float)
        x0 = spm_vec(X)

        # closure evaluates f at vectorised state
        #def eval_f(xv: np.ndarray) -> np.ndarray:
        #    #Xv = xv.reshape((ns, npop, nk), order="F")
        #    Xv = spm_unvec(xv, x_template)
        #    return tc_hilge2(Xv, u, P, M, return_jacobian=False, return_delay=False)[0]
        def eval_f(xv: np.ndarray) -> np.ndarray:
            Xv = spm_unvec(xv, x_template)
            return tc_hilge2(Xv, u, P, M, return_jacobian=False, return_delay=False)[0]

        for k in range(n):
            dx = np.zeros((n,), dtype=float)
            dx[k] = eps
            fp = eval_f(x0 + dx)
            fm = eval_f(x0 - dx)
            J[:, k] = (fp - fm) / (2.0 * eps)

    if return_delay:
        # Delay matrix logic from MATLAB tc_hilge2.m
        # Cortico-thalamic (CT) and thalamo-cortical (TC) delays (ms -> s)
        CT_ms = 8.0
        TC_ms = 3.0
        CT = CT_ms * float(np.exp(np.asarray(P.get("CT", 0.0), dtype=float)))
        TC = TC_ms * float(np.exp(np.asarray(P.get("TC", 0.0), dtype=float)))

        Tc = np.zeros((npop, npop), dtype=float)
        Tc[np.ix_([6, 7], list(range(0, 6)))] = CT
        Tc[np.ix_(list(range(0, 6)), [6, 7])] = TC
        Tc = Tc / 1000.0

        # Kronecker to state-space (npop*nk*ns)
        #Tc_big = np.kron(np.ones((nk, nk)), np.kron(Tc, np.ones((ns, ns))))
        Tc_big = np.kron(
            np.eye(ns),
            np.kron(Tc, np.eye(nk))
        )

        # Intra-population delays
        ID_base = np.array([2, 1, 1, 1, 1, 2, 1, 2], dtype=float)
        ID_scale = float(np.exp(np.asarray(P.get("ID", 0.0), dtype=float)))
        id_vec = np.repeat(ID_base * ID_scale / 1000.0, nk)  # length npop*nk
        IDmat = np.tile(id_vec.reshape(1, -1), (npop * nk, 1))
        #ID_big = np.kron(IDmat, np.ones((ns, ns)))
        ID_big = np.kron(
            np.eye(ns),
            np.diag(id_vec)
        )

        D = Tc_big + ID_big

    if not return_jacobian:
        J = None
    if not return_delay:
        D = None
    return f_vec, J, D

__all__ = ["tc_hilge2", "spm_vec", "spm_unvec", "reshape_x"]


