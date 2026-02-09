import numpy as np
from .laplace.transfer import laplace_tf
from .inference.vl_gc import fit_variational_laplace_thermo_gc


def _is_number(x):
    return isinstance(x, (int, float, np.number))

def _as_float_array(x):
    return np.asarray(x, dtype=float)

def pack_P0_general(P0: dict, include_keys=None, exclude_keys=None):
    """
    Flatten (pack) all numeric parameters in P0 into a single vector m0.

    - Supports scalars, np arrays, lists/tuples of arrays (e.g., A: [ns×ns]*5)
    - Skips non-numeric entries by default.
    - Returns (m0, meta, P_base) where meta stores how to unpack.

    include_keys: optional iterable; if provided, only these keys are considered.
    exclude_keys: optional iterable; keys to skip.
    """
    P_base = dict(P0)

    if include_keys is not None:
        include_keys = set(include_keys)
    if exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)

    # Stable ordering for reproducibility
    keys = sorted(P_base.keys())

    specs = []  # list of unpack specs in order
    flat_parts = []

    for k in keys:
        if k in exclude_keys:
            continue
        if include_keys is not None and k not in include_keys:
            continue

        v = P_base[k]

        # scalar numeric
        if _is_number(v):
            arr = np.array([float(v)], dtype=float)
            specs.append({"key": k, "kind": "scalar", "shape": (), "size": 1})
            flat_parts.append(arr)
            continue

        # numpy array numeric
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            arr = _as_float_array(v).reshape(-1)
            specs.append({"key": k, "kind": "ndarray", "shape": tuple(v.shape), "size": arr.size})
            flat_parts.append(arr)
            continue

        # list/tuple of numeric arrays (e.g. A/AN are lists)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            ok = True
            shapes = []
            sizes = []
            parts = []
            for item in v:
                if _is_number(item):
                    a = np.array([float(item)], dtype=float)
                    shapes.append(())
                    sizes.append(1)
                    parts.append(a)
                else:
                    aitem = np.asarray(item)
                    if not (isinstance(aitem, np.ndarray) and np.issubdtype(aitem.dtype, np.number)):
                        ok = False
                        break
                    a = _as_float_array(aitem).reshape(-1)
                    shapes.append(tuple(aitem.shape))
                    sizes.append(a.size)
                    parts.append(a)

            if ok:
                arr = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
                specs.append({
                    "key": k, "kind": "list",
                    "shapes": shapes, "sizes": sizes, "size": int(arr.size)
                })
                flat_parts.append(arr)
                continue

        # everything else: skip (flags, strings, None, dicts, etc.)
        # You can include dict recursion later if needed.
        # print(f"Skipping non-numeric key {k} type {type(v)}")

    m0 = np.concatenate(flat_parts) if flat_parts else np.zeros(0, dtype=float)
    meta = {"specs": specs, "keys_order": keys}
    return m0, meta, P_base


def unpack_P_general(m: np.ndarray, meta: dict, P_base: dict):
    """
    Reconstruct a P dict from vector m using pack_P0_general meta.
    """
    P = dict(P_base)
    m = np.asarray(m, dtype=float).ravel()
    idx = 0

    for spec in meta["specs"]:
        k = spec["key"]
        kind = spec["kind"]
        size = spec["size"]

        chunk = m[idx:idx+size]
        idx += size

        if kind == "scalar":
            P[k] = float(chunk[0])
        elif kind == "ndarray":
            P[k] = chunk.reshape(spec["shape"])
        elif kind == "list":
            out_list = []
            j = 0
            for sh, sz in zip(spec["shapes"], spec["sizes"]):
                part = chunk[j:j+sz]; j += sz
                if sh == ():
                    out_list.append(float(part[0]))
                else:
                    out_list.append(part.reshape(sh))
            P[k] = out_list
        else:
            raise ValueError(f"Unknown kind: {kind}")

    return P

def pC_zeros_like_P(P0: dict):
    """Return a dict with same keys/shapes as P0 but all zeros (numeric only)."""
    pC = {}
    for k, v in P0.items():
        if isinstance(v, (int, float, np.number)):
            pC[k] = 0.0
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            pC[k] = np.zeros_like(v, dtype=float)
        elif isinstance(v, (list, tuple)):
            out = []
            ok = True
            for item in v:
                if isinstance(item, (int, float, np.number)):
                    out.append(0.0)
                else:
                    a = np.asarray(item)
                    if not (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number)):
                        ok = False
                        break
                    out.append(np.zeros_like(a, dtype=float))
            if ok:
                pC[k] = out
        # non-numeric keys are skipped intentionally
    return pC

def expand_masked_to_full(m_small, m0_full, mask):
    """
    Insert the fitted (masked) parameters back into the full parameter vector.

    m_small : (k,) vector of free parameters
    m0_full : (d,) full parameter vector baseline (e.g. packed P0)
    mask    : (d,) boolean array, True where parameters are free
    """
    m_full = np.asarray(m0_full, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    m_full[mask] = np.asarray(m_small, dtype=float).reshape(-1)
    return m_full


def forward_spectrum(theta_dict, freqs, M):
    """Compute predicted cross-spectral density magnitude using Laplace TF."""
    M = dict(M)
    M['Hz'] = np.asarray(freqs, dtype=float)
    CSD, aux = laplace_tf(theta_dict, M)
    return CSD, aux

def default_J(ns: int, NPOP: int, NSTATES: int) -> np.ndarray:
    wV = np.array([0.2000, 0.9900, 0.1000, 0.8000, 0.1000, 0.2000, 0.0500, 0.1000], dtype=float)

    J = np.full((ns, NPOP, NSTATES), -20.0, dtype=float)   # exp(-20) ~ 2e-9
    J[:, :, 0] = np.log(np.maximum(wV[None, :], 1e-12))
    return J.reshape(-1, order="F")


def pack_params(P0: dict, ns: int, NPOP: int, NSTATES: int):
    n_total = ns * NPOP * NSTATES
    P0 = dict(P0)

    if "J" not in P0 or np.asarray(P0["J"]).size != n_total:
        P0["J"] = default_J(ns, NPOP, NSTATES)

    # baselines for scaled params
    H0  = np.asarray(P0["H"])
    Hn0 = np.asarray(P0["Hn"])
    Gb0 = float(P0.get("Gb", 1.0))

    m0 = np.concatenate([
        np.array([float(P0.get("leak", -2.0))], dtype=float),
        np.array([float(P0.get("L", 0.0))], dtype=float),
        np.asarray(P0["J"], dtype=float).reshape(-1),
        np.array([0.0, 0.0, 0.0], dtype=float),  # log-gains for H,Hn,Gb
    ])

    meta = dict(ns=ns, NPOP=NPOP, NSTATES=NSTATES, n_total=n_total, H0=H0, Hn0=Hn0, Gb0=Gb0)
    return m0, meta, P0


def unpack_params(m: np.ndarray, P0: dict, meta: dict) -> dict:
    ns, NPOP, NSTATES, n_total = meta["ns"], meta["NPOP"], meta["NSTATES"], meta["n_total"]
    H0, Hn0, Gb0 = meta["H0"], meta["Hn0"], meta["Gb0"]

    m = np.asarray(m, dtype=float).reshape(-1)
    i = 0
    leak = float(m[i]); i += 1
    L    = float(m[i]); i += 1
    J    = m[i:i+n_total].copy(); i += n_total
    gH, gHn, gGb = m[i:i+3].tolist()

    P = dict(P0)
    P["leak"] = leak
    P["L"]    = L
    P["J"]    = J
    P["H"]    = np.asarray(H0)  * np.exp(gH)
    P["Hn"]   = np.asarray(Hn0) * np.exp(gHn)
    P["Gb"]   = float(Gb0) * np.exp(gGb)
    return P

def vector_from_P_using_meta(P: dict, meta: dict):
    """
    Flatten P into a vector that matches the layout defined by meta["specs"]
    from pack_P0_general.
    """
    parts = []

    for spec in meta["specs"]:
        k = spec["key"]
        kind = spec["kind"]

        if k not in P:
            raise KeyError(f"Key '{k}' missing from P when vectorising.")

        v = P[k]

        if kind == "scalar":
            parts.append(np.array([float(v)], dtype=float))

        elif kind == "ndarray":
            arr = np.asarray(v, dtype=float)
            parts.append(arr.reshape(-1))

        elif kind == "list":
            # spec has: shapes (list), sizes (list)
            if not isinstance(v, (list, tuple)):
                raise TypeError(f"Key '{k}' expected list/tuple but got {type(v)}")

            if len(v) != len(spec["sizes"]):
                raise ValueError(
                    f"Key '{k}' list length mismatch: got {len(v)} expected {len(spec['sizes'])}"
                )

            chunk_parts = []
            for item, sh, sz in zip(v, spec["shapes"], spec["sizes"]):
                if sh == ():
                    chunk_parts.append(np.array([float(item)], dtype=float))
                else:
                    a = np.asarray(item, dtype=float)
                    # optional: sanity-check expected size
                    if a.size != sz:
                        a = a.reshape(-1)
                        if a.size != sz:
                            raise ValueError(f"Key '{k}' item size mismatch: got {a.size} expected {sz}")
                    chunk_parts.append(a.reshape(-1))

            parts.append(np.concatenate(chunk_parts) if chunk_parts else np.zeros(0, dtype=float))

        else:
            raise ValueError(f"Unknown spec kind: {kind}")

    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)



def fit_spectrum(freqs, Syy, P0, priors, M, opts=None):
    """Fit parameters to observed spectrum (masked by MATLAB-style pC nonzeros)."""
    if opts is None:
        opts = {}
    if priors is None:
        priors = {}

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    Syy = np.asarray(Syy)

    # --- observed log-spectrum vector
    if Syy.ndim == 1:
        y_obs = np.log(np.maximum(Syy, 1e-16))
    else:
        diag = np.stack([Syy[:, i, i] for i in range(Syy.shape[1])], axis=1)
        y_obs = np.log(np.maximum(diag.reshape(-1), 1e-16))

    P0_local = dict(P0)

    # ---- exclude flags by default (important)
    exclude = set(priors.get("exclude_keys", ["endo", "global", "scale_NMDA"]))

    # ---- pack full P0 once (this defines the layout)
    m0_full, meta, P0_local = pack_P0_general(P0_local, exclude_keys=exclude)

    # ---- build pC_like matching P0_local shapes
    pC_like = pC_zeros_like_P(P0_local)

    # Helper: expand 8x8 -> 8x8xns if needed
    def _expand_88_to_88ns(A, ns):
        A = np.asarray(A, dtype=float)
        if A.ndim == 2 and ns == 1:
            return A[:, :, None]
        elif A.ndim == 2 and ns > 1:
            return np.repeat(A[:, :, None], ns, axis=2)
        return A

    ns = int(M.get("ns", 1))

    # ---- MATLAB pC patterns you pasted (H/Hn + CT/TC + L)
    H_mat = np.array([
        [1,0,0,0,0,0,0,1],
        [1,1,1,0,0,0,0,0],
        [0,1,1,0,0,0,0,0],
        [0,1,0,1,1,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,1,0],
    ], dtype=float)

    Hn_mat = np.array([
        [0,0,0,0,0,0,0,1],
        [1,1,1,0,0,0,0,0],
        [0,1,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,0],
    ], dtype=float)

    # Only set keys that exist in P0_local (robust to struct differences)
    if "H" in pC_like:
        pC_like["H"] = _expand_88_to_88ns(H_mat, ns) if np.asarray(P0_local["H"]).ndim == 3 else H_mat
    if "Hn" in pC_like:
        pC_like["Hn"] = _expand_88_to_88ns(Hn_mat, ns) if np.asarray(P0_local["Hn"]).ndim == 3 else Hn_mat

    if "CT" in pC_like:
        pC_like["CT"] = 1.0
    if "TC" in pC_like:
        pC_like["TC"] = 1.0
    if "L" in pC_like:
        pC_like["L"] = 1.0
    
    if "d" in pC_like:
        pC_like["d"] = [1,1,1]

    # ---- flatten pC_like aligned with packing order
    pC_flat = vector_from_P_using_meta(pC_like, meta)

    # ---- mask: nonzero pC entries are free
    mask = pC_flat > 0.0
    if not np.any(mask):
        raise ValueError("pC mask has no nonzero entries after alignment. Check shapes/keys.")

    # ---- reduced parameter vector + diagonal priors from pC values
    m0 = m0_full[mask]
    S0_diag = pC_flat[mask].astype(float)

    pc_scale = float(priors.get("pC_scale", 1.0))
    S0_diag = np.maximum(S0_diag * pc_scale, 1e-12)
    S0 = np.diag(S0_diag)

    print("Packed full parameter count:", m0_full.size)
    print("Free parameters from pC mask:", m0.size)

    # ---- forward model uses masked vector, expands to full, then unpacks
    def f_model(m_small):
        m_full = expand_masked_to_full(m_small, m0_full, mask)
        P = unpack_P_general(m_full, meta, P0_local)

        CSD, _ = forward_spectrum(P, freqs, M)
        if CSD.ndim == 3:
            diag = np.stack([CSD[:, i, i] for i in range(CSD.shape[1])], axis=1)
            yhat = np.log(np.maximum(diag.reshape(-1), 1e-16))
        else:
            yhat = np.log(np.maximum(CSD.reshape(-1), 1e-16))
        return yhat

    # ---- run VL in masked space
    m_post_small, V_small, D, logL, iters, sigma2, allm, allF = fit_variational_laplace_thermo_gc(
        y_obs,
        f_model,
        m0,
        S0,
        max_iter=opts.get('max_iter', 64),
        tol=opts.get('tol', 1e-4),
        opts=opts,
    )

    # ---- expand posterior mean back to full, then unpack to parameter dict
    m_post_full = expand_masked_to_full(m_post_small, m0_full, mask)
    P_post = unpack_P_general(m_post_full, meta, P0_local)

    # ---- embed covariance back to full space (optional but nice)
    idx = np.where(mask)[0]
    d_full = m0_full.size
    V_full = np.zeros((d_full, d_full), dtype=float)
    V_full[np.ix_(idx, idx)] = V_small

    # ---- predicted spectrum with posterior parameters
    CSD_hat, aux = forward_spectrum(P_post, freqs, M)

    return dict(
        posterior=dict(
            mean=m_post_full,
            cov=V_full,
            mean_masked=m_post_small,
            cov_masked=V_small,
            mask=mask,
        ),
        params=P_post,
        predicted=dict(CSD=CSD_hat, aux=aux),
        diagnostics=dict(
            free_energy=allF,
            allm=allm,   # note: stored in masked space by VL
            logL=logL,
            iters=iters,
            sigma2=sigma2,
        ),
    )




