"""Antenna beam pattern."""
import numpy as np


def beam_gain(offset_deg, cfg):
    """One-way power gain vs boresight (1.0 = dead centre). Gaussian main lobe + flat sidelobe floor."""
    d = (np.asarray(offset_deg, dtype=float) + 180.0) % 360.0 - 180.0   # wrap to -180..180
    main = np.exp(-4 * np.log(2) * (d / cfg.beamwidth_deg) ** 2)         # -3 dB at half the beamwidth
    return np.maximum(main, 10 ** (cfg.sidelobe_db / 10))
