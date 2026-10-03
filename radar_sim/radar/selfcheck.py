"""Headless sanity tests: does the radar measure what we put into it?   python main.py --selfcheck"""
import numpy as np
from .config import RadarConfig
from .system import RadarSystem
from .targets import Target


def _clean_system(targets, **cfg_kw):
    cfg = RadarConfig(swerling=False, **cfg_kw)         # no fading, so results are repeatable
    s = RadarSystem(cfg, seed=3, targets=targets)
    s.env.clutter_on = False
    s.env.jammer_on = False
    return s


def _target_at(name, R, az, vr, rcs):
    a = np.radians(az)
    return Target(name, R * np.sin(a), R * np.cos(a), vr * np.sin(a), vr * np.cos(a), rcs)


def check_range_and_velocity():
    print("1) range + velocity accuracy (pulse-Doppler, no clutter)")
    ok = True
    for R, vr in [(9000, 120), (4000, -200), (12500, 35)]:
        t = _target_at("t", R, 40, vr, 20)
        s = _clean_system([t])
        s.stare_az = 40
        dets = s._stare()
        if not dets:
            print(f"   truth R={R} v={vr:+}  -> NOT DETECTED"); ok = False; continue
        r, v, snr = dets[0]
        dv = s.cfg.wavelength * s.cfg.prf / (2 * s.cfg.doppler_pulses)
        good = abs(r - R) <= s.cfg.range_resolution and abs(v - vr) <= dv
        ok &= good
        print(f"   truth R={R:5d} m v={vr:+4d} m/s  ->  measured R={r:7.1f} m v={v:+6.1f} m/s  SNR {snr:4.1f} dB  {'ok' if good else 'FAIL'}")
    return ok


def check_false_alarm_rate():
    print("2) CFAR false alarm rate (noise only, pfa set to 1e-4)")
    s = _clean_system([], pfa_rd=1e-4)
    cells = crossings = 0
    for _ in range(80):
        s._stare()
        crossings += s.rd_raw_crossings
        cells += s.rd_db.shape[0] * (s.cfg.n_samples - s.cfg.blind_bins)
    rate = crossings / cells
    good = 0.4e-4 < rate < 2.5e-4
    print(f"   expected 1.0e-04, measured {rate:.2e} over {cells:,} cells  {'ok' if good else 'FAIL'}")
    return good


def check_mti_and_clutter():
    print("3) MTI with ground clutter: stationary mast should vanish, mover should stay")
    mast = _target_at("mast", 6000, 100, 0, 100)
    mover = _target_at("mover", 9000, 100, -150, 20)
    s = _clean_system([mast, mover])
    s.env.clutter_on = True
    def seen(R, dets): return any(abs(d[0] - R) < 60 for d in dets)
    s.mti_on = False
    off = s._ppi_dwell(100)
    s.mti_on = True
    on = s._ppi_dwell(100)
    results = [("mast, MTI off", seen(6000, off), True), ("mover, MTI off", seen(9000, off), True),
               ("mast, MTI on", seen(6000, on), False), ("mover, MTI on", seen(9000, on), True)]
    good = True
    for label, got, want in results:
        good &= (got == want)
        print(f"   {label:15s} detected={got!s:5s} (want {want!s:5s}) {'ok' if got == want else 'FAIL'}")
    return good


def run():
    results = [check_range_and_velocity(), check_false_alarm_rate(), check_mti_and_clutter()]
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
    return all(results)
