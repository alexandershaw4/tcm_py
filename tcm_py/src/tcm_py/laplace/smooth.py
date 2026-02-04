import numpy as np

def agauss_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    """Approx Gaussian smoothing along first axis using FFT convolution.
    sigma is in same units as sample spacing (assumed 1). Caller should scale.
    """
    y = np.asarray(y)
    if sigma <= 0:
        return y
    n = y.shape[0]
    # frequency grid
    k = np.fft.fftfreq(n)
    # Gaussian in Fourier domain: exp(-2*pi^2*sigma^2*k^2)
    g = np.exp(-2*(np.pi**2)*(sigma**2)*(k**2))
    Y = np.fft.fft(y, axis=0)
    return np.real(np.fft.ifft(Y * g[:, None] if y.ndim>1 else Y*g, axis=0))
