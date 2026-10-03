"""The 'air' between radar and world: builds the raw echoes the receiver would hear for one burst of pulses."""
import numpy as np
from .config import C
from .antenna import beam_gain


class Channel:
    def __init__(self, cfg, wf, env, seed=100):
        self.cfg, self.wf, self.env = cfg, wf, env
        self.rng = np.random.default_rng(seed)

    def _cnoise(self, shape, power):
        z = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        return np.sqrt(power / 2) * z

    def dwell(self, targets, az_deg, n_pulses):
        """Fire n_pulses at az_deg. Returns the pulse-compressed data, shape (pulses, range bins)."""
        cfg, wf, n = self.cfg, self.wf, self.cfg.n_samples
        tm = np.arange(n_pulses) / cfg.prf                        # when each pulse leaves
        rx = self._cnoise((n_pulses, n), cfg.noise_power)         # receiver noise is always there
        rows = np.arange(n_pulses)[:, None] * np.ones(wf.n, dtype=int)

        for tg in targets:
            R0 = tg.range
            if R0 < cfg.blind_range or R0 > cfg.max_range:
                continue
            # radar equation: how much power bounces back
            p = (cfg.peak_power * cfg.gain ** 2 * cfg.wavelength ** 2 * tg.rcs
                 * float(beam_gain(tg.azimuth_deg - az_deg, cfg)) ** 2) / ((4 * np.pi) ** 3 * R0 ** 4)
            if p < cfg.noise_power * 1e-3:
                continue                                           # way below the noise, don't waste time
            amp = np.sqrt(p)
            if cfg.swerling:                                       # random amplitude + phase, same for the whole burst
                amp = amp * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2)
            Rm = R0 + tg.radial_velocity * tm                      # range at each pulse (target keeps moving)
            td = 2 * Rm / C                                        # round trip delay
            idx = np.ceil(td * cfg.fs).astype(int)[:, None] + np.arange(wf.n)
            tt = idx / cfg.fs - td[:, None]                        # time since the echo's leading edge
            # the carrier phase changes by 4*pi/lambda per metre -> that's the Doppler shift across pulses
            echo = amp * np.exp(-4j * np.pi * Rm / cfg.wavelength)[:, None] * wf.echo_at(tt)
            ok = idx < n
            rx[rows[ok], idx[ok]] += echo[ok]

        if self.env.jammer_on:                                     # barrage noise jammer, different every sample
            rx += self._cnoise((n_pulses, n), self.env.jammer_power_at(az_deg))

        clutter = None
        if self.env.clutter_on:
            wobble = np.cumsum(self.rng.normal(0, 0.03, n_pulses))  # leaves/waves jiggle -> what limits MTI
            clutter = self.env.clutter_at(az_deg)[None, :] * np.exp(1j * wobble)[:, None]

        pc = wf.compress(rx, clutter)
        pc[:, :cfg.blind_bins] = 0                                 # receiver was off while transmitting
        return pc
