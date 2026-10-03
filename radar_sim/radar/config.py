"""Every radar number lives here, so you can redesign the whole radar from one file."""
import math
from dataclasses import dataclass

C = 299_792_458.0       # speed of light, m/s
K_B = 1.380649e-23      # boltzmann constant, J/K


@dataclass
class RadarConfig:
    # --- RF + waveform (S-band, roughly an air traffic control radar) ---
    fc: float = 3.0e9              # carrier frequency, Hz
    bandwidth: float = 5.0e6       # chirp sweep -> 30 m range resolution
    pulse_width: float = 10e-6     # chirp length, s
    fs: float = 10.0e6             # complex baseband sample rate, Hz
    prf: float = 10_000.0          # pulses per second -> 15 km max range, +-250 m/s max speed

    # --- transmitter / antenna / receiver ---
    peak_power: float = 1_000.0    # W
    gain_db: float = 25.0          # antenna gain
    beamwidth_deg: float = 3.0     # half power beamwidth
    sidelobe_db: float = -35.0     # far sidelobe level (this is how jammers sneak in)
    noise_figure_db: float = 3.0
    temperature_k: float = 290.0
    rpm: float = 12.0              # antenna rotation, one lap every 5 s

    # --- processing ---
    ppi_pulses: int = 16           # pulses per beam position when scanning
    doppler_pulses: int = 64       # pulses per coherent interval in pulse-Doppler mode
    mti_order: int = 2             # 1 = two-pulse canceller, 2 = three-pulse
    pfa_ppi: float = 1e-6          # false alarm probability per range cell
    pfa_rd: float = 1e-7           # same, for the range-Doppler map
    ppi_margin_db: float = 10.0    # never call something a target unless it's this far above its surroundings
    doppler_notch_bins: int = 1    # ignore detections this close to zero velocity (that's just the ground)
    rd_margin_db: float = 8.0      # (stationary clutter breaks CFAR's maths, this margin keeps the false alarms sane)

    # --- the world ---
    clutter_cnr_db: float = 8.0    # clutter-to-noise at 5 km, before any patches
    jammer_jnr_db: float = 42.0    # jammer-to-noise when the beam points right at it
    jammer_az_deg: float = 130.0
    swerling: bool = True          # targets fade in and out like real aircraft

    # ---------- derived stuff ----------
    @property
    def wavelength(self): return C / self.fc
    @property
    def gain(self): return 10 ** (self.gain_db / 10)
    @property
    def n_samples(self): return int(round(self.fs / self.prf))        # range bins per pulse
    @property
    def range_bin(self): return C / (2 * self.fs)                      # metres per sample
    @property
    def max_range(self): return self.n_samples * self.range_bin
    @property
    def range_resolution(self): return C / (2 * self.bandwidth)
    @property
    def blind_bins(self): return int(round(self.pulse_width * self.fs))  # deaf while transmitting
    @property
    def blind_range(self): return self.blind_bins * self.range_bin
    @property
    def unambiguous_velocity(self): return self.wavelength * self.prf / 4
    @property
    def noise_power(self):
        return K_B * self.temperature_k * 10 ** (self.noise_figure_db / 10) * self.fs
