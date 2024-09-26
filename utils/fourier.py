from scipy.fft import fft, fftfreq
import numpy as np

def compute_fourier(spike_train, sampling_rate):
    n = len(spike_train)
    yf = fft(spike_train)
    xf = fftfreq(n, 1 / sampling_rate)[:n // 2]
    return xf, 2.0 / n * np.abs(yf[:n // 2])