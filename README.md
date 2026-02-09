# 🧠 tcm_py

**Thalamo-Cortical Neural Mass Modelling in Python**

A Python implementation of an extended conductance-based
thalamo-cortical neural mass model (TCM), designed for computational
psychiatry, neuropharmacology, and generative modelling applications.

This package provides:

-   A biologically grounded thalamo-cortical neural mass model
-   Laplace-domain transfer function computation
-   Jacobian and delay handling
-   Compatibility with Variational Laplace / Thermodynamic VL
-   A clean Python interface for model inversion and simulation

Developed within the Computational Psychiatry & Neuropharmacological
Systems (CPNS) Lab. 
https://cpnslab.com

------------------------------------------------------------------------

## 🔬 Scientific Context

This model implements an extended canonical thalamo-cortical neural mass
architecture inspired by Dynamic Causal Modelling (DCM), predictive
coding, and conductance-based population models.

Populations per cortical source:

1.  Spiny Stellate (L4)
2.  Superficial Pyramidal (L2/3)
3.  Superficial Interneurons
4.  Deep Pyramidal (L5)
5.  Deep Interneurons
6.  Thalamic Projection Pyramidal (L6)
7.  Reticular Thalamus
8.  Thalamic Relay Cells

Conductances modelled:

-   AMPA
-   NMDA
-   GABA-A
-   GABA-B
-   Optional M-channel
-   Optional H-channel

The implementation supports:

-   Linearisation around fixed points
-   Laplace-domain spectral predictions
-   Cross-spectral density reconstruction
-   Delay-aware dynamics
-   Parameter estimation via Variational Laplace

------------------------------------------------------------------------

## 📦 Installation

Clone the repository:

``` bash
git clone https://github.com/yourusername/tcm_py.git
cd tcm_py
```

Create a virtual environment (recommended):

``` bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Minimal dependencies typically include:

-   numpy
-   scipy
-   matplotlib

------------------------------------------------------------------------

## 🏗 Project Structure

    tcm_py/
    │
    ├── tcm/
    │   ├── tc_hilge2.py
    │   ├── single_source_dynamics.py
    │   ├── compute_delays.py
    │   ├── compute_jacobian.py
    │   ├── laplace_transfer_function.py
    │
    ├── inference/
    │   ├── fit_variational_laplace_thermo.py
    │   ├── fixed_point_solver.py
    │
    ├── examples/
    │   ├── test.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🚀 Quick Start

### 1️⃣ Simulate the model

``` python
from tcm.tc_hilge2 import tc_hilge2
import numpy as np

P = {...}
M = {...}
x = np.zeros(M["n_states"])
u = np.zeros(M["n_inputs"])

dx = tc_hilge2(x, u, P, M)
```

------------------------------------------------------------------------

### 2️⃣ Compute Laplace-domain transfer function

``` python
from tcm.laplace_transfer_function import laplace_transfer_function
import numpy as np

w = np.linspace(1, 100, 128)
G = laplace_transfer_function(P, M, w)
```

------------------------------------------------------------------------

### 3️⃣ Fit parameters with Thermodynamic Variational Laplace

``` python
from inference.fit_variational_laplace_thermo import fit_variational_laplace_thermo

m, V, diagnostics = fit_variational_laplace_thermo(
    y,
    forward_model,
    m0,
    S0
)
```

------------------------------------------------------------------------

## 🧠 Model Inversion Philosophy

This package is built around generative modelling principles:

Forward model:\
y = f(θ) + ε

Variational Laplace approximation:\
Gaussian posterior over parameters

Thermodynamic integration:\
Robust free energy estimation

The design supports:

-   Spectral DCM-style inversion
-   Pharmacological perturbation modelling
-   Synaptic parameter estimation
-   E/I balance inference
-   Precision modulation analysis

------------------------------------------------------------------------

## 🔁 Fixed Point Solver

The model linearises around a steady-state operating point.

If you see:

⚠️ Fixed point solver did not fully converge

This typically means:

-   Parameter scaling needs adjustment
-   Initial state guess is poor
-   Conductance values are too extreme

Check:

-   Leak conductance (GL)
-   Synaptic gains
-   External input scaling

------------------------------------------------------------------------

## 📈 Applications

This framework is suitable for:

-   EEG/MEG spectral modelling
-   Dynamic Causal Modelling
-   Computational psychiatry
-   Neuropharmacology
-   Active inference agents (TCM as world model)
-   Generative modelling research

------------------------------------------------------------------------

## 🔮 Roadmap

Planned extensions:

-   Multi-source connectivity matrices
-   Empirical Bayes group inversion
-   Polyphonic Variational Laplace integration
-   Mixture noise models
-   Julia interoperability
-   GPU acceleration

------------------------------------------------------------------------

## 📚 References

If you use this code in academic work, please cite and link.

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests welcome.

Please:

-   Add docstrings
-   Include minimal working examples
-   Avoid breaking API without version bump

------------------------------------------------------------------------

## 📜 License

MIT License 
