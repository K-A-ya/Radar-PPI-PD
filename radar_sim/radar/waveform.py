"""The chirp we transmit, and the matched filter that squeezes it back into a sharp spike."""
import numpy as np


class Waveform:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = int(round(cfg.pulse_width * cfg.fs))             # samples in one chirp
        self.k = cfg.bandwidth / cfg.pulse_width                   # chirp rate, Hz per second
        t = (np.arange(self.n) - self.n / 2) / cfg.fs
        self.tx = np.exp(1j * np.pi * self.k * t ** 2)             # LFM chirp, complex baseband
        # hamming taper on the replica = low range sidelobes, costs about 1.3 dB of SNR
        self.replica = self.tx * np.hamming(self.n)
        self.n_fft = 1 << int(np.ceil(np.log2(cfg.n_samples + self.n)))  # big enough that nothing wraps
        self.S = np.fft.fft(self.tx, self.n_fft)
        self.H = np.conj(np.fft.fft(self.replica, self.n_fft))     # matched filter = conjugate of the replica
        self.noise_gain = float(np.mean(np.abs(self.H) ** 2))       # power gain for plain white noise
        self.clutter_gain = float(np.mean(np.abs(self.S * self.H) ** 2))  # for stuff that got chirped first

    def echo_at(self, tt):
        """Chirp value tt seconds after its leading edge (caller keeps 0 <= tt < pulse_width)."""
        return np.exp(1j * np.pi * self.k * (tt - self.cfg.pulse_width / 2) ** 2)

    def compress(self, rx, clutter=None):
        """Pulse compression, done in the frequency domain (fast convolution).
        clutter = reflectivity that still needs to be 'chirped' (its echo = reflectivity convolved with the pulse)."""
        X = np.fft.fft(rx, self.n_fft, axis=-1)
        if clutter is not None:
            X += np.fft.fft(clutter, self.n_fft, axis=-1) * self.S
        return np.fft.ifft(X * self.H, axis=-1)[..., :self.cfg.n_samples]
