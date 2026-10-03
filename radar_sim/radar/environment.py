"""The world around the radar: ground clutter and a noise jammer."""
import numpy as np
from .antenna import beam_gain


class Environment:
    def __init__(self, cfg, wf, seed=7):
        self.cfg = cfg
        self.clutter_on = True
        self.jammer_on = True
        self.jam_power = cfg.noise_power * 10 ** (cfg.jammer_jnr_db / 10)
        rng = np.random.default_rng(seed)
        self.clutter_map = self._make_clutter(wf, rng)
        # the beam is wider than 1 degree, so mix neighbouring azimuth cells by the beam pattern
        self._k = np.arange(-6, 7)
        w = beam_gain(self._k, cfg) ** 2
        self._w = np.sqrt(w / w.sum())

    def _make_clutter(self, wf, rng):
        cfg, n = self.cfg, self.cfg.n_samples
        R = np.maximum((np.arange(n) + 0.5) * cfg.range_bin, 500.0)
        az = np.arange(360)[:, None]
        cnr = 10 ** (cfg.clutter_cnr_db / 10) * (5000.0 / R) ** 3        # ground clutter falls off like 1/R^3
        def blob(az0, r0, saz, sr, gain_db):                               # a smooth bump of extra reflectivity
            d = (az - az0 + 180) % 360 - 180
            return (10 ** (gain_db / 10) - 1) * np.exp(-(d / saz) ** 2 - ((R - r0) / sr) ** 2)
        patch = 1 + blob(57, 6500, 10, 1200, 16) + blob(205, 3000, 8, 700, 12) + blob(300, 9500, 15, 1500, 10)  # hills, a city, more hills
        cnr = cnr * patch
        # convert 'CNR after the matched filter' back into reflectivity variance at the antenna
        var = cfg.noise_power * wf.noise_gain * cnr / wf.clutter_gain
        z = rng.standard_normal((360, n)) + 1j * rng.standard_normal((360, n))
        return np.sqrt(var / 2) * z

    def clutter_at(self, az_deg):
        """Complex reflectivity per range cell seen by a beam pointing at az_deg."""
        a = int(round(az_deg)) % 360
        return (self._w[:, None] * self.clutter_map[(a + self._k) % 360]).sum(axis=0)

    def jammer_power_at(self, az_deg):
        return self.jam_power * float(beam_gain(az_deg - self.cfg.jammer_az_deg, self.cfg))
