import numpy as np

def fd_jacobian(fun, x, eps=1e-6):
    """Finite-difference Jacobian of fun(x) returning vector."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x))
    J = np.zeros((f0.size, x.size), dtype=float)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        fp = np.asarray(fun(xp))
        fm = np.asarray(fun(xm))
        J[:, i] = (fp - fm) / (2*eps)
    return J
