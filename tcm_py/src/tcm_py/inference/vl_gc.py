import numpy as np
from scipy import linalg

def build_gc_operators(T: int, q: int, dt: float = 1.0):
    """Finite-difference operator lifting a vector into generalised coordinates.
    Returns a list of operators D^k for k=0..q mapping R^T -> R^T.
    """
    I = np.eye(T)
    ops = [I]
    # first derivative (central)
    D1 = np.zeros((T, T))
    for t in range(1, T-1):
        D1[t, t+1] = 0.5/dt
        D1[t, t-1] = -0.5/dt
    # boundaries: forward/backward
    D1[0,0] = -1/dt; D1[0,1]=1/dt
    D1[-1,-2] = -1/dt; D1[-1,-1]=1/dt
    ops.append(D1)
    for k in range(2, q+1):
        ops.append(D1 @ ops[k-1])
    return ops

def lift_gc(v: np.ndarray, ops):
    v = v.reshape(-1,1)
    return np.vstack([op @ v for op in ops]).reshape(-1)

def fit_variational_laplace_thermo_gc(y, f, m0, S0, max_iter=64, tol=1e-4, opts=None):
    """Port of fitVariationalLaplaceThermo_GC.m (simplified + Pythonic).

    y : (T,) data vector (or flattened)
    f : callable(m) -> yhat OR (yhat, J) where J = dyhat/dm
    m0 : (d,)
    S0 : (d,d)
    opts: dict with keys:
        q: GC order (default 0)
        dt: spacing for FD operator (default 1)
        beta_schedule: list/array of beta (thermo), default [1]
        sigma2: initial noise variance (default 1e-2)
        lam: GN damping (default 0)
        boundary: optional (lo, hi) arrays for parameters
    """
    if opts is None:
        opts = {}
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.asarray(m0, dtype=float).reshape(-1)
    S0 = np.asarray(S0, dtype=float)

    T = y.size
    q = int(opts.get('q', 0))
    dt = float(opts.get('dt', 1.0))
    ops = build_gc_operators(T, q, dt)

    beta_schedule = np.asarray(opts.get('beta_schedule', [1.0]), dtype=float)
    sigma2 = float(opts.get('sigma2', 1e-2))
    lam = float(opts.get('lam', 0.0))

    lohi = opts.get('boundary', None)

    allm = []
    all_elbo = []

    min_iter = int(opts.get("min_iter", 5))

    # prior precision
    iS0 = linalg.inv(S0)

    def apply_boundary(m):
        if lohi is None:
            return m
        lo, hi = lohi
        return np.minimum(np.maximum(m, lo), hi)

    V = S0.copy()

    for beta in beta_schedule:
        for it in range(max_iter):
            out = f(m)
            if isinstance(out, tuple) and len(out) == 2:
                yhat, J = out
            else:
                yhat = out
                # finite-diff jacobian
                eps = 1e-6
                d = m.size
                J = np.zeros((T, d))
                for i in range(d):
                    mp = m.copy(); mp[i]+=eps
                    mm = m.copy(); mm[i]-=eps
                    J[:, i] = (np.asarray(f(mp)).reshape(-1) - np.asarray(f(mm)).reshape(-1)) / (2*eps)

            yhat = np.asarray(yhat, dtype=float).reshape(-1)
            r = y - yhat

            # lift to GC
            r_gc = lift_gc(r, ops)
            J_gc = np.vstack([op @ J for op in ops])

            # likelihood precision
            iSv = (beta / sigma2) * np.eye(r_gc.size)

            # Laplace/GN update
            A = iS0 + J_gc.T @ iSv @ J_gc + lam*np.eye(m.size)
            b = iS0 @ (m0 - m) + J_gc.T @ iSv @ r_gc

            #dm = linalg.solve(A, b, assume_a='pos')
            try:
                dm = linalg.solve(A, b, assume_a='pos')
            except Exception:
                A = A + 1e-3*np.eye(m.size)
                dm = linalg.solve(A, b)
            m_new = apply_boundary(m + dm)

            # free-energy-ish (for monitoring)
            quad = float(r_gc.T @ (iSv @ r_gc))
            prior_quad = float((m_new-m0).T @ (iS0 @ (m_new-m0)))
            F = -0.5*(quad + prior_quad)
            all_elbo.append(F)
            allm.append(m_new.copy())

            #if np.linalg.norm(dm) < tol:
            #    m = m_new
            #    break

            if it + 1 >= min_iter and np.linalg.norm(dm) < tol:
                m = m_new
                break
            m = m_new

        # posterior covariance at this beta
        V = linalg.inv(A)

    logL = all_elbo[-1] if all_elbo else np.nan
    iter_total = len(all_elbo)

    return m, V, None, logL, iter_total, sigma2, np.array(allm), np.array(all_elbo)
