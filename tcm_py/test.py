import numpy as np
import matplotlib.pyplot as plt

from tcm_py import fit_spectrum
from tcm_py.dcm import build_dcm

# --------------------------------
# Observed spectrum
# --------------------------------
#freqs = np.linspace(1.0, 80.0, 100)
#psd = 1.0 / (freqs ** 1.5)

freqs = np.linspace(1,80,100)
psd = 1/(freqs ** 1.5) + 2*np.exp( -(10 - freqs) **2 / (2*(2*4)**2)) + np.exp( -(50 - freqs) **2 / (2*(2*4)**2))
plt.figure()
plt.plot(freqs, psd, label="Observed")
plt.show()

# --------------------------------
# Build model
# --------------------------------
P0, M = build_dcm(ns=1, freqs=freqs, find_fp=True)


from tcm_py.api import forward_spectrum

# get your internal parameter vector m0 from wherever you pack it
# easiest: grab it from inside fit_spectrum once you expose it, but for now:
P = P0.copy()
CSD0, _ = forward_spectrum(P, freqs, M)

# poke a parameter that MUST matter
P2 = P0.copy()
P2["leak"] = P0.get("leak", -2.0) + 0.5
CSD1, _ = forward_spectrum(P2, freqs, M)

d = np.linalg.norm(np.real(CSD1).ravel() - np.real(CSD0).ravel())
print("delta spectrum from leak poke:", d)


# --------------------------------
# Fit
# --------------------------------
out = fit_spectrum(
    freqs=freqs,
    Syy=psd,
    P0=P0,
    priors={"S0": 1.0},
    M=M,
    opts={"q": 0, "max_iter": 32, "tol": 1e-5, "sigma2": 1e-2},
)

# --------------------------------
# Plot
# --------------------------------
CSD = np.asarray(out["predicted"]["CSD"])
Shat = np.real(CSD[:, 0, 0]) if CSD.ndim == 3 else np.real(CSD)

plt.figure()
plt.plot(freqs, psd, label="Observed")
plt.plot(freqs, Shat, label="Predicted")
plt.yscale("log")
plt.legend()
plt.show()

plt.figure()
plt.plot(out["diagnostics"]["free_energy"])
plt.title("Free Energy Trace")
plt.show()

diag = out.get("diagnostics", {})
print("diagnostics keys:", list(diag.keys()))

F = diag.get("F_trace", None)
if F is None:
    # try other quick candidates
    for k in ["F", "Fhist", "allF", "elbo_trace", "free_energy"]:
        if k in diag:
            F = diag[k]
            print("Using key:", k)
            break

print("F_trace type:", type(F))
if F is not None:
    try:
        print("F_trace length:", len(F))
        print("F_trace head:", list(F)[:5])
    except Exception as e:
        print("Could not inspect F_trace:", e)

print("iters:", diag.get("iters", out.get("iters", None)))
print("sigma2:", diag.get("sigma2", out.get("sigma2", None)))

print("params keys:", list(out.get("params", {}).keys()))

