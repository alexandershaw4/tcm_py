import numpy as np
from scipy import linalg
from ..utils.linalg import denan
from .smooth import agauss_smooth

def laplace_tf(P, M, U=None):
    """Port of Alex_LaplaceTFwDNew.m.

    Parameters
    ----------
    P : dict
        Parameters (may be vectorised elsewhere; here expect dict-like).
    M : dict
        Must contain:
            - 'Hz': frequency vector (F,)
            - 'x': state template (ns×np×nk) OR vectorised
            - 'f': callable model function with signature f(x,u,P,M, return_jacobian=True, return_delay=True)
    U : dict or None
        Not used in this port; kept for parity.

    Returns
    -------
    CSD : ndarray
        (F, ns, ns) cross-spectral (magnitude-smoothed).
    aux : dict
        Contains PSD (ns,F) complex response, MAG/PHA per region etc.
    """
    if isinstance(P, dict) and 'p' in P:
        P = P['p']

    if U is None:
        U = np.zeros((1,), dtype=float)

    w = np.asarray(M['Hz'], dtype=float).reshape(-1)
    x0 = np.asarray(M['x'], dtype=float).reshape(-1, order='F')

    # Determine endogenous/exogenous
    endogenous = bool(M.get('endogenous', False))
    Input = 0 if endogenous else 1

    # Linearisation: f0, A, D
    model_fun = M['f']
    # Expect model_fun returns (f, A, D) when requested
    f0, A, D = model_fun(M['x'], 0.0, P, M, return_jacobian=True, return_delay=True)
    A = denan(A)
    D = denan(D)

    # Input Jacobian wrt u (finite diff on u)
    # In MATLAB: Bfull = spm_diff(M.f,M.x,1,P,M,2)
    # Here: numerical derivative w.r.t scalar u
    eps = 1e-6
    #f_plus = model_fun(M['x'], eps, P, M)
    #f_minus = model_fun(M['x'], -eps, P, M)
    #Bfull = ((f_plus - f_minus) / (2*eps)).reshape(-1, 1)
    f_plus = M["f"](M["x"], U + eps, P, M)[0]
    f_minus = M["f"](M["x"], U - eps, P, M)[0]
    Bfull = ((f_plus - f_minus) / (2 * eps)).reshape(-1, 1)

    # external spectrum
    Uomega = np.ones_like(w)
    if 'external_spectrum' in M:
        Uomega = np.asarray(M['external_spectrum'], dtype=float).reshape(-1)

    damp = 0.0
    if 'd' in P and len(np.atleast_1d(P['d'])) >= 2:
        damp = float(np.exp(np.atleast_1d(P['d'])[1]))

    # assume ns sources equals first dim of M['x']
    x_template = np.asarray(M['x'])
    ns = x_template.shape[0]

    PSD = np.zeros((ns, w.size), dtype=complex)
    MAG = []
    PHA = []

    for ii in range(ns):
        win = np.arange(ii, A.shape[0], ns)   # every ns-th state (0-based)
        n = win.size

        AA = A[np.ix_(win, win)]
        BB = Bfull[win, :]  # (n,1)

        # observer weights: exp(P.J(win)) in MATLAB
        if 'J' in P:
            Cw = np.exp(np.asarray(P['J']).reshape(-1, order='F')[win])
        else:
            Cw = np.ones(n)
        X0 = x0[win]

        drive_scale = 1.0
        if 'C' in P and len(np.atleast_1d(P['C'])) >= (ii+1):
            drive_scale = float(np.exp(np.atleast_1d(P['C'])[ii]))

        # collapse multi-input columns if any (here BB is single col)
        BB_eff = BB.reshape(n)

        MG = np.zeros((n, w.size), dtype=complex)
        y = np.zeros((w.size,), dtype=complex)

        for j, hz in enumerate(w):
            s = damp + 1j*2*np.pi*hz
            # delays: elementwise exp(-j*2*pi*w*D)
            E = np.exp(-1j*2*np.pi*hz * D[np.ix_(win, win)])
            Aef = AA * E
            Jm = (s*np.eye(n, dtype=complex)) - Aef

            if Input:
                u_j = Uomega[j] * drive_scale
                Ym = linalg.solve(Jm, BB_eff*u_j, assume_a='gen') + linalg.solve(Jm, X0, assume_a='gen')
            else:
                Ym = linalg.solve(Jm, X0, assume_a='gen')
            MG[:, j] = Ym
            y[j] = np.dot(Cw.conj().T, Ym)

        if M.get('ham', False):
            Hm = np.hamming(w.size)
            y = y * Hm

        MAG.append(MG)
        PHA.append(np.angle(MG) * 180/np.pi)

        Lgain = 1.0
        if 'L' in P and len(np.atleast_1d(P['L'])) >= (ii+1):
            Lgain = float(np.exp(np.atleast_1d(P['L'])[ii]))

        Y = y.copy()
        # curvature penalty: Y = Y - (exp(P.d(1))*3)*H with H = grad(grad(Y))
        if 'd' in P and len(np.atleast_1d(P['d'])) >= 1:
            d1 = float(np.exp(np.atleast_1d(P['d'])[0]))
            H = np.gradient(np.gradient(Y))
            Y = Y - (d1*3.0)*H

        PSD[ii, :] = Lgain * Y

    # Cross-spectra heuristic (as in MATLAB)
    CSD = np.zeros((w.size, ns, ns), dtype=complex)
    for i in range(ns):
        CSD[:, i, i] = PSD[i, :]
        for j in range(ns):
            if i == j:
                continue
            Lc = 1.0
            if 'Lc' in P and len(np.atleast_1d(P['Lc'])) >= (i+1):
                Lc = float(np.exp(np.atleast_1d(P['Lc'])[i]))
            if Input:
                CSD[:, i, j] = Lc * (PSD[i, :] * np.conj(PSD[j, :]))
            else:
                CSD[:, i, j] = Lc * (PSD[i, :] * (PSD[j, :]))
            CSD[:, j, i] = CSD[:, i, j]

    # Smooth magnitudes
    dw = float(np.mean(np.diff(w))) if w.size > 1 else 1.0
    sigma = 0.0
    if 'd' in P and len(np.atleast_1d(P['d'])) >= 3:
        sigma = dw * float(np.exp(np.atleast_1d(P['d'])[2]))

    CSD_abs = np.abs(CSD)
    if sigma > 0:
        for i in range(ns):
            for j in range(ns):
                CSD_abs[:, i, j] = agauss_smooth(CSD_abs[:, i, j], sigma)

    aux = dict(PSD=PSD, MAG=MAG, PHA=PHA, x0=x0, dx=f0)
    return CSD_abs, aux
