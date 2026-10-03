"""The radar's brain: spins the antenna, runs the processing chain, remembers what it saw."""
from collections import deque
import numpy as np
from .config import RadarConfig
from .waveform import Waveform
from .environment import Environment
from .channel import Channel
from .targets import default_scenario
from . import processing as dsp

PPI_GUARD, PPI_TRAIN = (3, 0), (16, 0)      # (range, doppler) cells on each side of the cell under test
RD_GUARD, RD_TRAIN = (3, 2), (10, 4)
PERSISTENCE = 2.5                            # seconds, how long the green phosphor glows


class RadarSystem:
    def __init__(self, cfg=None, seed=1, targets=None):
        self.cfg = cfg or RadarConfig()
        self.wf = Waveform(self.cfg)
        self.env = Environment(self.cfg, self.wf, seed=seed)
        self.channel = Channel(self.cfg, self.wf, self.env, seed=seed + 100)
        self.targets = default_scenario() if targets is None else targets
        n = self.cfg.n_samples
        self.mode = "ppi"
        self.mti_on = False
        self.time = 0.0
        self._az_total = 0.0                                  # antenna angle, never wrapped (easier maths)
        self.ppi_img = np.zeros((360, n), np.float32)         # phosphor brightness per (azimuth, range)
        self.hits = deque(maxlen=400)                         # [time, x, y, snr] for the scope
        self.profile_db = np.zeros(n)                         # latest range profile (A-scope)
        self.threshold_db = None
        self.rd_db = None                                     # latest range-Doppler map
        self.rd_axis = None                                   # velocity of each Doppler row
        self.rd_dets = []                                     # (range, velocity, snr)
        self.rd_raw_crossings = 0
        self.stare_az = self.targets[0].azimuth_deg if self.targets else 0.0
        self._aim = -1
        self._alphas = {}

    # ---------- helpers ----------
    @property
    def az(self): return self._az_total % 360.0

    def _alpha(self, pfa, guard, train, n_int=1):
        key = (pfa, guard, train, n_int)
        if key not in self._alphas:
            self._alphas[key] = dsp.cfar_alpha(pfa, dsp.n_training_cells(guard, train), n_int)
        return self._alphas[key]

    def set_mode(self, mode):
        self.mode = mode
        if mode == "doppler":
            self._stare()

    def aim_at_next_target(self):
        self._aim = (self._aim + 1) % len(self.targets)
        self.stare_az = round(self.targets[self._aim].azimuth_deg) % 360
        return self.stare_az

    # ---------- time ----------
    def advance(self, dt):
        self.time += dt
        for t in self.targets:
            t.step(dt)
        self.ppi_img *= np.float32(np.exp(-dt / PERSISTENCE))
        if self.mode == "ppi":
            old = self._az_total
            self._az_total += self.cfg.rpm * 6.0 * dt           # rpm * 360 deg / 60 s
            for b in range(int(np.floor(old)) + 1, int(np.floor(self._az_total)) + 1):
                self._ppi_dwell(b % 360)                         # one burst for every degree we sweep past
        else:
            self._stare()

    # ---------- PPI: scan, MTI (optional), non-coherent integration, 1D CFAR ----------
    def _ppi_dwell(self, b):
        cfg, bl = self.cfg, self.cfg.blind_bins
        y = self.channel.dwell(self.targets, float(b), cfg.ppi_pulses)
        norm = self.wf.noise_gain * cfg.noise_power              # what pure noise looks like after the filter
        m_eff = cfg.ppi_pulses
        if self.mti_on:
            y = dsp.mti(y, cfg.mti_order)
            norm *= dsp.mti_noise_gain(cfg.mti_order)
            m_eff = max(1, (cfg.ppi_pulses - cfg.mti_order) // 2)   # MTI outputs are correlated, so count fewer pulses = safer threshold
        p = (np.abs(y) ** 2).mean(axis=0) / norm                  # average the pulses' power (non-coherent integration)
        # the margin is a floor on the threshold: stationary clutter is the same on every pulse, so averaging pulses
        # doesn't smooth it out like CFAR's maths assumes -> without a floor it'd light up with false alarms
        alpha = max(self._alpha(cfg.pfa_ppi, PPI_GUARD, PPI_TRAIN, m_eff), 10 ** (cfg.ppi_margin_db / 10))
        valid = p[None, bl:]                                      # skip the deaf zone, CFAR would choke on those zeros
        hit, thr = dsp.ca_cfar(valid, PPI_GUARD, PPI_TRAIN, alpha)
        hit = dsp.local_peaks(valid, hit)[0]
        thr = thr[0]

        db = dsp.to_db(p)
        self.ppi_img[b] = np.clip(db / 40.0, 0, 1)
        self.profile_db = db
        self.threshold_db = np.concatenate([np.full(bl, np.nan), dsp.to_db(thr)])
        out = []
        for j in np.flatnonzero(hit):
            R = (bl + j) * cfg.range_bin
            snr = float(dsp.to_db(valid[0, j] / (thr[j] / alpha)))
            self._add_hit(R * np.sin(np.radians(b)), R * np.cos(np.radians(b)), snr)
            out.append((R, float(b), snr))
        return out

    def _add_hit(self, x, y, snr):
        for h in list(self.hits)[-8:]:                            # same target seen by neighbouring beam positions? merge
            if self.time - h[0] < 0.6 and (h[1] - x) ** 2 + (h[2] - y) ** 2 < 300.0 ** 2:
                if snr > h[3]:
                    h[1:4] = [x, y, snr]
                h[0] = self.time
                return
        self.hits.append([self.time, x, y, snr])

    # ---------- pulse-Doppler: stare, MTI (optional), Doppler FFT, 2D CFAR ----------
    def _stare(self):
        cfg, bl = self.cfg, self.cfg.blind_bins
        y = self.channel.dwell(self.targets, self.stare_az, cfg.doppler_pulses)
        norm = self.wf.noise_gain * cfg.noise_power
        if self.mti_on:
            y = dsp.mti(y, cfg.mti_order)
            norm *= dsp.mti_noise_gain(cfg.mti_order)
        M = y.shape[0]
        w = np.hamming(M)                                         # window the pulses so strong echoes don't smear across all Doppler bins
        spec = np.fft.fftshift(np.fft.fft(y * w[:, None], axis=0), axes=0)   # FFT across pulses = Doppler
        p = np.abs(spec) ** 2 / (norm * np.sum(w ** 2))           # noise-only cells now average exactly 1
        fd = np.fft.fftshift(np.fft.fftfreq(M, 1 / cfg.prf))      # Doppler frequency of each row
        v = -cfg.wavelength * fd / 2                              # fd = -2 v / lambda, so + means moving away
        p, v = p[::-1], v[::-1]                                   # flip so velocity increases left to right

        alpha = max(self._alpha(cfg.pfa_rd, RD_GUARD, RD_TRAIN, 1), 10 ** (cfg.rd_margin_db / 10))
        valid = p[:, bl:]
        hit, thr = dsp.ca_cfar(valid, RD_GUARD, RD_TRAIN, alpha)
        self.rd_raw_crossings = int(hit.sum())
        hit = dsp.local_peaks(valid, hit)
        hit[np.abs(v) <= cfg.doppler_notch_bins * (v[1] - v[0]) * 1.001] = False   # zero-Doppler notch: the ground sits here

        db = dsp.to_db(p)
        db[:, :bl] = 0
        self.rd_db, self.rd_axis = db, v
        self.profile_db = db.max(axis=0)
        self.threshold_db = None
        self.rd_dets = [((bl + j) * cfg.range_bin, float(v[i]), float(dsp.to_db(valid[i, j] / (thr[i, j] / alpha))))
                        for i, j in np.argwhere(hit)]
        self.rd_dets.sort(key=lambda d: -d[2])
        return self.rd_dets
