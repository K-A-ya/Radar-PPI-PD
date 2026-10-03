"""Things that fly around and reflect radio waves."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Target:
    name: str
    x: float          # metres east of the radar
    y: float          # metres north of the radar
    vx: float         # m/s
    vy: float
    rcs: float        # radar cross section, m^2 (how big it looks to radar, not how big it is)

    @property
    def range(self): return float(np.hypot(self.x, self.y))
    @property
    def azimuth_deg(self): return float(np.degrees(np.arctan2(self.x, self.y)) % 360)  # clockwise from north
    @property
    def radial_velocity(self):
        """Positive = moving AWAY from the radar, negative = coming toward it."""
        return (self.x * self.vx + self.y * self.vy) / self.range

    def step(self, dt, r_min=2500.0, r_max=14000.0):
        self.x += self.vx * dt
        self.y += self.vy * dt
        r, vr = self.range, self.radial_velocity
        if (r > r_max and vr > 0) or (r < r_min and vr < 0):      # leaving the arena? bounce back in
            ux, uy = self.x / r, self.y / r
            self.vx -= 2 * vr * ux
            self.vy -= 2 * vr * uy


def default_scenario():
    return [
        Target("Airliner",   -9000,  6000,  210,  -40,  60),   # fast, big, coming toward us
        Target("Jet",        11000,  3000,  -60,  150,  40),   # crosses sideways (almost no radial speed)
        Target("Fighter",     5000, -8000, -200,  150,   4),   # fast, small
        Target("Cessna",      3500,  4000,  -35,  -20,   3),
        Target("Helicopter", -5000, -3000,   12,    5,   8),   # crawling, MTI will struggle to see it
        Target("Drone",       2500,  2000,  -30,  -20, 0.05),  # tiny, hides in clutter
        Target("Radio mast",     0, -6500,    0,    0, 100),   # stationary: MTI deletes it
    ]
