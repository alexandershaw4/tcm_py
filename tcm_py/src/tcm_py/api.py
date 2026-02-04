import numpy as np
from .laplace.transfer import laplace_tf
from .inference.vl_gc import fit_variational_laplace_thermo_gc

def forward_spectrum(theta_dict, freqs, M):
    """Compute predicted cross-spectral density magnitude using Laplace TF."""
    M = dict(M)
    M['Hz'] = np.asarray(freqs, dtype=float)
    CSD, aux = laplace_tf(theta_dict, M)
    return CSD, aux

def fit_spectrum(freqs, Syy, P0, priors, M, opts=None):
    """Fit parameters to observed spectrum.

    Parameters
    ----------
    freqs : (F,)
    Syy : (F,) or (F, ns, ns) observed spectral magnitude
    P0 : dict initial parameters (same structure used by laplace_tf/model)
    priors : dict with 'm0','S0' for parameter vectorisation (prototype: only supports scalar leak)
    M : dict model structure
    opts : dict, passed to VL_GC

    Notes
    -----
    This prototype currently fits a single scalar parameter P['leak'] (log-space)
    to demonstrate the end-to-end pipeline. Extend vectorisation to full P struct
    once tc_hilge2 port is completed.
    """
    if opts is None:
        opts = {}
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    Syy = np.asarray(Syy)
    if Syy.ndim == 1:
        y_obs = np.log(np.maximum(Syy, 1e-16))
    else:
        # fit to diagonal autospectra stacked
        diag = np.stack([Syy[:,i,i] for i in range(Syy.shape[1])], axis=1)
        y_obs = np.log(np.maximum(diag.reshape(-1), 1e-16))

    # Parameterisation: m is scalar = P['leak'] in log-space
    m0 = np.array([float(P0.get('leak', -2.0))], dtype=float)
    S0 = np.array([[float(priors.get('S0', 1.0))]], dtype=float)

    def f_model(m):
        P = dict(P0)
        P['leak'] = float(m[0])
        CSD, _ = forward_spectrum(P, freqs, M)
        if CSD.ndim == 3:
            diag = np.stack([CSD[:,i,i] for i in range(CSD.shape[1])], axis=1)
            yhat = np.log(np.maximum(diag.reshape(-1), 1e-16))
        else:
            yhat = np.log(np.maximum(CSD.reshape(-1), 1e-16))
        return yhat

    m_post, V, D, logL, iters, sigma2, allm, allF = fit_variational_laplace_thermo_gc(
        y_obs, f_model, m0, S0, max_iter=opts.get('max_iter',64), tol=opts.get('tol',1e-4), opts=opts
    )

    P_post = dict(P0); P_post['leak'] = float(m_post[0])

    CSD_hat, aux = forward_spectrum(P_post, freqs, M)

    return dict(
        posterior=dict(mean=m_post, cov=V),
        params=P_post,
        predicted=dict(CSD=CSD_hat, aux=aux),
        diagnostics=dict(free_energy=allF, allm=allm, logL=logL, iters=iters)
    )
