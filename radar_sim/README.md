# Radar simulator

A simulated S-band pulse radar. Targets reflect real chirps through the radar equation, the receiver
adds noise, ground clutter and a jammer, and then real signal processing finds the targets again.
Nothing is faked on the display: every blip is a CFAR detection from the simulated echoes.

    pip install -r requirements.txt
    python main.py               # live scope
    python main.py --selfcheck   # headless accuracy tests (range, velocity, false alarm rate, MTI)

## Using the scope
| control | what it does |
|---|---|
| mode radio / `m` | **PPI scope** (rotating antenna) or **Pulse-Doppler** (antenna stares, you get a range-velocity map) |
| Ground clutter | hills + a city that reflect at zero speed |
| Noise jammer | barrage jammer at 130 degrees, shows up as a wedge |
| MTI filter | three-pulse canceller, deletes anything stationary |
| CFAR hits | yellow circles in PPI, green in range-Doppler |
| Show truth | red x = where targets really are, plus a truth table |
| stare az / aim button | where the beam points in Pulse-Doppler mode |
| `space` | pause |

## Things to try
* **MTI on in PPI mode.** The radio mast and the hills vanish, and so do the Jet (flying across the beam),
  the Helicopter and the Drone (too slow along the line of sight). MTI only passes targets with radial speed.
* **Switch to Pulse-Doppler and aim at the Helicopter.** Doppler processing separates it from the ground
  (zero speed) even though MTI just threw it away.
* **Aim at the Radio mast.** It sits in the zero-velocity stripe and is deliberately not declared a target.
* **Jammer off/on.** In the main beam it blinds a whole wedge; its far sidelobes lift the noise everywhere.

## How the code is laid out
    radar/config.py       every number (frequency, bandwidth, PRF, power...) + derived values like range resolution
    radar/waveform.py     LFM chirp + Hamming-weighted matched filter (pulse compression)
    radar/antenna.py      beam pattern: Gaussian main lobe + sidelobe floor
    radar/targets.py      moving targets, bounce off arena edges
    radar/environment.py  clutter map (1/R^3 + hills) and jammer
    radar/channel.py      builds raw echoes: radar equation, delays, Doppler phase, noise, clutter, jamming
    radar/processing.py   MTI, exact CA-CFAR threshold, fast 2D CFAR (summed-area table), peak picking
    radar/system.py       antenna scan, PPI chain, Doppler chain, detection bookkeeping
    radar/display.py      matplotlib scope + widgets
    radar/selfcheck.py    headless tests

## The signal chain
    targets/clutter/jammer/noise -> raw echoes -> matched filter (30 m resolution)
        PPI:      [MTI] -> average 16 pulses -> 1D CFAR -> detections
        Doppler:  [MTI] -> Hamming window -> FFT over 64 pulses -> 2D CFAR -> notch zero velocity -> detections

Key numbers: 5 MHz chirp = 30 m resolution, 15 km max range, 10 kHz PRF = unambiguous speed +-250 m/s,
Doppler bin = 7.8 m/s, receiver is deaf for the first 1.5 km while the pulse is going out.

## Honest simplifications
* "Stop and hop": targets don't move during one pulse, and we ignore range-Doppler coupling of the chirp.
* Flat earth, no multipath, no atmosphere, constant antenna gain model, one jammer.
* CA-CFAR assumes noise-like surroundings. Stationary clutter breaks that, so the PPI/Doppler detectors also
  enforce a minimum margin above the local average (see `ppi_margin_db`, `rd_margin_db` in `config.py`).
  The measured noise-only false alarm rate lands within about 2x of the design value.


## thank you for reading the whole docuent

## :)