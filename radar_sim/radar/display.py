"""The live scope: a green-phosphor PPI, a range-Doppler map, an A-scope and some controls."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

BG, FG, DIM, ACCENT = "#04100a", "#a6e3b4", "#2f6b44", "#f5d76e"
PHOSPHOR = LinearSegmentedColormap.from_list("phosphor", ["#000000", "#003a12", "#00a830", "#7dff8c", "#eaffea"])
MODES = ["PPI scope", "Pulse-Doppler"]
CHECKS = ["Ground clutter", "Noise jammer", "MTI filter", "CFAR hits", "Show truth"]


class RadarApp:
    FRAME_DT = 0.04       # seconds of radar time per animation frame
    HIT_LIFE = 8.0        # how long a detection marker stays on the PPI
    PIX = 560             # PPI image resolution

    def __init__(self, system):
        self.sys = system
        self.paused = False
        self.mode_idx = 0
        self.show_hits, self.show_truth = True, False
        plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                             "axes.edgecolor": DIM, "font.size": 9})
        self.fig = plt.figure(figsize=(13, 7.4), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Radar simulator")
        self._build_ppi()
        self._build_rd()
        self._build_side()
        self._build_controls()
        self._apply_mode()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(self.fig, self._frame, interval=int(self.FRAME_DT * 1000), cache_frame_data=False)

    # ---------------- building the scope ----------------
    def _build_ppi(self):
        cfg, R = self.sys.cfg, self.sys.cfg.max_range
        ax = self.ax_ppi = self.fig.add_axes([0.01, 0.02, 0.63, 0.96], facecolor="black")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-R * 1.08, R * 1.08); ax.set_ylim(-R * 1.08, R * 1.08)

        # lookup tables: for every screen pixel, which (azimuth, range) cell does it show? built once, used every frame
        xs = np.linspace(-R, R, self.PIX)
        X, Y = np.meshgrid(xs, xs)
        rr = np.hypot(X, Y)
        self.lut_az = np.minimum((np.degrees(np.arctan2(X, Y)) % 360).astype(int), 359)
        self.lut_r = np.minimum((rr / cfg.range_bin).astype(int), cfg.n_samples - 1)
        self.lut_mask = rr < R
        self.im_ppi = ax.imshow(np.zeros((self.PIX, self.PIX)), extent=[-R, R, -R, R], origin="lower",
                                cmap=PHOSPHOR, vmin=0, vmax=1, interpolation="bilinear", zorder=1)

        for r in range(3000, int(R) + 1, 3000):                                    # range rings
            ax.add_patch(Circle((0, 0), r, fill=False, ec=DIM, lw=0.8, zorder=2))
            ax.text(r * 0.7071 + 120, r * 0.7071 + 120, f"{r // 1000} km", color=DIM, fontsize=8, zorder=2)
        for a in range(0, 360, 30):                                                # compass spokes
            s, c = np.sin(np.radians(a)), np.cos(np.radians(a))
            ax.plot([0, R * s], [0, R * c], color=DIM, lw=0.5, zorder=2)
            ax.text(R * 1.05 * s, R * 1.05 * c, f"{a:03d}", color=FG, ha="center", va="center", fontsize=8)
        self.sweep, = ax.plot([0, 0], [0, R], color="#c9ffd2", lw=1.6, alpha=0.9, zorder=3)
        self.hit_sc = ax.scatter([], [], s=110, facecolors="none", linewidths=1.6, zorder=5)
        self.truth_sc = ax.scatter([], [], s=45, marker="x", c="#ff6b6b", linewidths=1.5, zorder=4)
        self.truth_sc.set_visible(False)
        ax.plot(0, 0, marker="+", color=ACCENT, ms=10, zorder=6)

    def _build_rd(self):
        cfg = self.sys.cfg
        ax = self.ax_rd = self.fig.add_axes([0.07, 0.08, 0.55, 0.85], facecolor="black")
        vmax = cfg.unambiguous_velocity
        self.im_rd = ax.imshow(np.zeros((cfg.n_samples, cfg.doppler_pulses)), aspect="auto", origin="lower",
                               cmap="inferno", vmin=0, vmax=50, extent=[-vmax, vmax, 0, cfg.max_range / 1000],
                               interpolation="nearest")
        ax.set_xlabel("radial velocity, m/s   (+ = moving away, - = coming toward us)")
        ax.set_ylabel("range, km")
        ax.set_title("Range-Doppler map (brightness = dB above noise)", color=FG, fontsize=10)
        ax.axvline(0, color=DIM, lw=0.6)
        self.rd_sc = ax.scatter([], [], s=150, facecolors="none", edgecolors="#6dff95", linewidths=2)

    def _build_side(self):
        cfg = self.sys.cfg
        ax = self.ax_scope = self.fig.add_axes([0.70, 0.76, 0.28, 0.19], facecolor="black")
        self.r_km = (np.arange(cfg.n_samples) + 0.5) * cfg.range_bin / 1000
        self.ln_prof, = ax.plot(self.r_km, np.zeros_like(self.r_km), color="#7dff8c", lw=1)
        self.ln_thr, = ax.plot(self.r_km, np.zeros_like(self.r_km), color=ACCENT, lw=1, ls="--")
        ax.set_xlim(0, cfg.max_range / 1000); ax.set_ylim(-5, 60)
        ax.set_xlabel("range, km"); ax.set_ylabel("dB over noise")
        ax.grid(color=DIM, lw=0.4)
        self.scope_title = ax.set_title("", color=FG, fontsize=9)
        self.info = self.fig.text(0.70, 0.71, "", va="top", family="monospace", fontsize=8, color=FG)

    def _build_controls(self):
        s = self.sys
        a = self.fig.add_axes([0.70, 0.14, 0.14, 0.15], facecolor=BG); a.set_frame_on(False)
        states = [s.env.clutter_on, s.env.jammer_on, s.mti_on, True, False]
        try:   # newer matplotlib lets us colour the ticks so they show on the dark background
            self.check = CheckButtons(a, CHECKS, states, frame_props={"edgecolor": FG}, check_props={"color": ACCENT})
        except TypeError:
            self.check = CheckButtons(a, CHECKS, states)
        self.check.on_clicked(self._on_check)
        a = self.fig.add_axes([0.85, 0.20, 0.13, 0.09], facecolor=BG); a.set_frame_on(False)
        try:
            self.radio = RadioButtons(a, MODES, active=0, radio_props={"facecolor": ACCENT, "edgecolor": FG})
        except TypeError:
            self.radio = RadioButtons(a, MODES, active=0)
        self.radio.on_clicked(self._on_mode)
        for lab in list(self.check.labels) + list(self.radio.labels):
            lab.set_color(FG); lab.set_fontsize(8.5)
        a = self.fig.add_axes([0.76, 0.095, 0.20, 0.025], facecolor=BG)
        self.slider = Slider(a, "stare az", 0, 359, valinit=s.stare_az, valstep=1, color=DIM)
        self.slider.on_changed(lambda v: setattr(s, "stare_az", float(v)))
        a = self.fig.add_axes([0.76, 0.035, 0.20, 0.04])
        self.btn = Button(a, "aim at next target", color="#12301f", hovercolor="#1d4b30")
        self.btn.label.set_color(FG)
        self.btn.on_clicked(lambda _e: self.slider.set_val(s.aim_at_next_target()))
        self.fig.text(0.70, 0.005, "space = pause    m = switch mode    q = quit", color=DIM, fontsize=8)

    # ---------------- events ----------------
    def _on_check(self, _label):
        st = dict(zip(CHECKS, self.check.get_status()))
        s = self.sys
        s.env.clutter_on, s.env.jammer_on, s.mti_on = st["Ground clutter"], st["Noise jammer"], st["MTI filter"]
        self.show_hits, self.show_truth = st["CFAR hits"], st["Show truth"]
        self.truth_sc.set_visible(self.show_truth and self.mode_idx == 0)

    def _on_mode(self, label):
        self.mode_idx = MODES.index(label)
        self.sys.set_mode("ppi" if self.mode_idx == 0 else "doppler")
        self._apply_mode()

    def _apply_mode(self):
        ppi = self.mode_idx == 0
        self.ax_ppi.set_visible(ppi)
        self.ax_rd.set_visible(not ppi)
        self.slider.ax.set_visible(not ppi)
        self.btn.ax.set_visible(not ppi)
        self.ln_thr.set_visible(ppi)
        self.truth_sc.set_visible(self.show_truth and ppi)

    def _on_key(self, e):
        if e.key == " ":
            self.paused = not self.paused
        elif e.key == "m":
            self.radio.set_active(1 - self.mode_idx)

    # ---------------- drawing ----------------
    def _frame(self, _i):
        if not self.paused:
            self.sys.advance(self.FRAME_DT)
        self._draw_ppi() if self.mode_idx == 0 else self._draw_rd()
        self._draw_scope()
        self._draw_info()

    def _draw_ppi(self):
        s, R = self.sys, self.sys.cfg.max_range
        self.im_ppi.set_data(s.ppi_img[self.lut_az, self.lut_r] * self.lut_mask)   # polar -> screen pixels
        a = np.radians(s.az)
        self.sweep.set_data([0, R * np.sin(a)], [0, R * np.cos(a)])
        live = [h for h in s.hits if s.time - h[0] < self.HIT_LIFE] if self.show_hits else []
        if live:
            xy = np.array([[h[1], h[2]] for h in live])
            alpha = np.clip(1 - np.array([s.time - h[0] for h in live]) / self.HIT_LIFE, 0.1, 1)
            col = np.tile(np.array([1.0, 0.85, 0.3, 1.0]), (len(live), 1)); col[:, 3] = alpha
            self.hit_sc.set_offsets(xy); self.hit_sc.set_edgecolors(col); self.hit_sc.set_visible(True)
        else:
            self.hit_sc.set_visible(False)
        if self.show_truth:
            self.truth_sc.set_offsets(np.array([[t.x, t.y] for t in s.targets]))

    def _draw_rd(self):
        s, cfg = self.sys, self.sys.cfg
        if s.rd_db is None:
            return
        v = s.rd_axis; dv = v[1] - v[0]
        self.im_rd.set_data(s.rd_db.T)                              # rows = range, columns = velocity
        self.im_rd.set_extent([v[0] - dv / 2, v[-1] + dv / 2, 0, cfg.max_range / 1000])
        d = s.rd_dets if self.show_hits else []
        self.rd_sc.set_offsets(np.array([[x[1], x[0] / 1000] for x in d]) if d else np.empty((0, 2)))

    def _draw_scope(self):
        s = self.sys
        self.ln_prof.set_ydata(np.maximum(s.profile_db, -5))
        if self.mode_idx == 0 and s.threshold_db is not None:
            self.ln_thr.set_ydata(s.threshold_db)
        self.scope_title.set_text(f"A-scope: azimuth {s.az:5.1f} deg" if self.mode_idx == 0
                                  else f"strongest Doppler per range, azimuth {s.stare_az:.0f} deg")

    def _draw_info(self):
        s, c = self.sys, self.sys.cfg
        ppi = self.mode_idx == 0
        L = [f"MODE  {MODES[self.mode_idx]}",
             f"AZ    {s.az:6.1f} deg" if ppi else f"STARE {s.stare_az:6.1f} deg",
             f"range res {c.range_resolution:.0f} m   max {c.max_range / 1000:.0f} km",
             f"v unamb +-{c.unambiguous_velocity:.0f} m/s   PRF {c.prf / 1000:.0f} kHz",
             f"MTI {'on ' if s.mti_on else 'off'}  Pfa {(c.pfa_ppi if ppi else c.pfa_rd):.0e}", ""]
        if ppi:
            L.append("latest detections (range, az, SNR)")
            for h in list(s.hits)[-(3 if self.show_truth else 6):][::-1]:
                L.append(f" {np.hypot(h[1], h[2]) / 1000:5.2f} km {np.degrees(np.arctan2(h[1], h[2])) % 360:5.1f}  {h[3]:4.1f} dB")
        else:
            L.append("detections (range, velocity, SNR)")
            for r, v, snr in s.rd_dets[:(4 if self.show_truth else 7)]:
                L.append(f" {r / 1000:5.2f} km {v:+7.1f} m/s {snr:4.1f} dB")
        if self.show_truth:
            L += ["", "truth (range, az, radial v)"]
            L += [f" {t.name[:10]:10s}{t.range / 1000:5.2f}km {t.azimuth_deg:5.1f} {t.radial_velocity:+6.1f}" for t in s.targets]
        self.info.set_text("\n".join(L))
