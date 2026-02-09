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
    """Port of fitVariationalLaplaceThermo_GC.m (simplified + Pythonic) with line-search.

    Notes
    -----
    - Adds backtracking line-search to prevent oscillatory accept/reject behaviour.
    - Computes a consistent monitoring objective F(m) using both residual and prior at the same m.
    """
    if opts is None:
        opts = {}

    # Ensure minimum GN damping for stability
    opts["lam"] = max(float(opts.get("lam", 0.0)), 1e-2)
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

    # line-search options
    ls_max = int(opts.get("ls_max", 12))
    ls_shrink = float(opts.get("ls_shrink", 0.5))
    ls_min_step = float(opts.get("ls_min_step", 1e-6))

    # prior precision
    iS0 = linalg.inv(S0)

    def apply_boundary(mv):
        if lohi is None:
            return mv
        lo, hi = lohi
        return np.minimum(np.maximum(mv, lo), hi)

    def eval_model_and_jac(mv):
        """Return yhat (T,), J (T,d) at mv."""
        out = f(mv)
        if isinstance(out, tuple) and len(out) == 2:
            yhat, J = out
            yhat = np.asarray(yhat, dtype=float).reshape(-1)
            J = np.asarray(J, dtype=float)
        else:
            yhat = np.asarray(out, dtype=float).reshape(-1)
            # finite-diff jacobian
            eps = 1e-6
            d = mv.size
            J = np.zeros((T, d), dtype=float)
            for i in range(d):
                mp = mv.copy(); mp[i] += eps
                mm = mv.copy(); mm[i] -= eps
                fp = np.asarray(f(mp)).reshape(-1)
                fm = np.asarray(f(mm)).reshape(-1)
                J[:, i] = (fp - fm) / (2 * eps)
        return yhat, J

    def compute_F_from_r_gc(r_gc, mv, iSv):
        """Monitoring objective (free-energy-ish): higher is better."""
        quad = float(r_gc.T @ (iSv @ r_gc))
        prior_quad = float((mv - m0).T @ (iS0 @ (mv - m0)))
        return -0.5 * (quad + prior_quad)

    V = S0.copy()

    for beta in beta_schedule:
        for it in range(max_iter):
            # evaluate at current m
            yhat, J = eval_model_and_jac(m)
            r = y - yhat

            # lift to GC
            r_gc = lift_gc(r, ops)
            J_gc = np.vstack([op @ J for op in ops])

            # likelihood precision
            iSv = (beta / sigma2) * np.eye(r_gc.size)

            # GN update (at current m)
            A = iS0 + J_gc.T @ iSv @ J_gc + lam * np.eye(m.size)
            b = iS0 @ (m0 - m) + J_gc.T @ iSv @ r_gc

            try:
                dm = linalg.solve(A, b, assume_a='pos')
            except Exception:
                A = A + 1e-3 * np.eye(m.size)
                dm = linalg.solve(A, b)

            # --- line search on dm, using consistent F(m)
            m_old = m.copy()
            F_old = compute_F_from_r_gc(r_gc, m_old, iSv)

            step = 1.0
            accepted = False
            m_new = m_old

            for _ in range(ls_max):
                m_try = apply_boundary(m_old + step * dm)

                # evaluate F at m_try (must recompute residual at that point)
                yhat_try, _ = eval_model_and_jac(m_try)  # jac not needed for acceptance
                r_try = y - yhat_try
                r_gc_try = lift_gc(r_try, ops)
                F_try = compute_F_from_r_gc(r_gc_try, m_try, iSv)

                if F_try > F_old:   # higher is better
                    m_new = m_try
                    F_new = F_try
                    accepted = True
                    break

                step *= ls_shrink
                if step < ls_min_step:
                    break

            if not accepted:
                # reject: stay at m_old
                m_new = m_old
                F_new = F_old

            all_elbo.append(F_new)
            allm.append(m_new.copy())

            # stopping (use actual taken step size)
            dm_taken = m_new - m_old
            m = m_new

            if it + 1 >= min_iter and np.linalg.norm(dm_taken) < tol:
                break

        # posterior covariance at this beta (from last A at current m)
        V = linalg.inv(A)

    logL = all_elbo[-1] if all_elbo else np.nan
    iter_total = len(all_elbo)

    return m, V, None, logL, iter_total, sigma2, np.array(allm), np.array(all_elbo)
