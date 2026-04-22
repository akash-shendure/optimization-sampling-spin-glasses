"""Arrhenius-wall animation: the same local sampler on the same rugged
landscape becomes qualitatively different as the water level drops.

    Scene1_ThreePanels -- three stacked energy-landscape panels at
        beta_high, beta_crit, beta_low. Each panel shows the same 1D
        rugged landscape from glass_transition with its own water level
        and its own accepted-hop trail from a single MH chain. Because
        the rng sequence is shared, the only thing driving the panels
        apart is the acceptance rule. Each panel's per-hop animation
        step_dt is tuned so all three panels finish at the same wall
        clock: the high-T panel sprints through many hops, the low-T
        panel crawls through a few -- hops fire at regular intervals
        within each panel rather than stopping and starting on dwells.

Render pipeline (matches the other animations in this repo):

    manim scene.py Scene1_ThreePanels
    ffmpeg -y -i Scene1_ThreePanels.mp4 -filter_complex \
        "[0:v]split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse" \
        -loop -1 Scene1_ThreePanels.gif

Videos land next to this file at 1080p / 50 fps.
"""
from pathlib import Path

import numpy as np
from manim import *

_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".manim_cache"
config.media_dir = str(_CACHE)
config.video_dir = str(_HERE)
config.partial_movie_dir = str(_CACHE / "partial_movie_files")
config.quality = "high_quality"
config.frame_rate = 50
config.background_color = WHITE

# ---------- palette ----------
FG = BLACK
WATER_FILL = BLUE_C
WATER_LINE = BLUE_E
PARTICLE = RED_E

EQN_SIZE = 38
LABEL_SIZE = 28

# ---------- layout ----------
# Three panels stacked vertically, centered now that there is no title.
X_MIN, X_MAX = -5.0, 5.0
PANEL_WIDTH = 10.2
PANEL_HEIGHT = 1.80
PANEL_Y = (2.55, 0.00, -2.55)   # top, middle, bottom centers
PANEL_X_SHIFT = 0.0             # panels centered; labels float to the left


# ===================================================================
# Densely rugged landscape -- same functional form as glass_transition
# so the animations are visually continuous across the deck.
# ===================================================================

def _envelope(x):
    return (
        -1.75 * np.exp(-((x - 2.00) / 0.90) ** 2)
        - 1.30 * np.exp(-((x + 2.25) / 1.00) ** 2)
        - 0.95 * np.exp(-((x - 0.00) / 0.75) ** 2)
        - 0.80 * np.exp(-((x + 4.00) / 0.55) ** 2)
    )


_MODES = (
    (0.85,  0.70, 0.32),
    (1.35,  2.10, 0.26),
    (1.90, -0.30, 0.22),
    (2.55,  1.50, 0.18),
    (3.25,  0.80, 0.15),
    (4.10, -0.90, 0.13),
    (5.00,  1.80, 0.11),
    (6.25,  0.20, 0.09),
    (7.70, -1.20, 0.08),
    (9.30,  0.50, 0.07),
    (11.0,  1.90, 0.06),
    (13.2, -0.60, 0.05),
    (15.5,  1.30, 0.045),
    (17.9, -0.40, 0.04),
    (20.3,  0.90, 0.035),
)


def _rugged(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * np.sin(f * x + p)
    return out


def energy(x):
    x = np.asarray(x, dtype=float)
    return _envelope(x) + _rugged(x)


def _energy_scalar(x):
    return float(energy(np.array([float(x)]))[0])


_PROBE_XS = np.linspace(X_MIN, X_MAX, 6000)
_PROBE_YS = energy(_PROBE_XS)
_E_MIN = float(_PROBE_YS.min())
_E_MAX = float(_PROBE_YS.max())

Y_MIN = _E_MIN - 0.40
Y_MAX = _E_MAX + 0.50

E_FLOOR = _E_MIN - 0.05
LEVEL_CAP = _E_MAX + 0.65
THERMAL_C = 1.60

BETA_HIGH = 0.35   # liquid, water above every peak
BETA_CRIT = 1.10   # critical, water between main barriers -- exposes several peaks
BETA_LOW = 2.60    # glassy, water just above deepest basin


def level_from_beta(beta):
    return float(min(E_FLOOR + THERMAL_C / max(float(beta), 1e-3), LEVEL_CAP))


# ===================================================================
# Axes and landscape rendering
# ===================================================================

def _panel_axes(y_center):
    ax = Axes(
        x_range=[X_MIN, X_MAX, 1],
        y_range=[Y_MIN, Y_MAX, 1],
        x_length=PANEL_WIDTH,
        y_length=PANEL_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 1.8, "include_ticks": False},
    )
    ax.move_to(np.array([PANEL_X_SHIFT, y_center, 0.0]))
    return ax


def _panel_curve(ax):
    return ax.plot(
        lambda x: _energy_scalar(x),
        x_range=[X_MIN, X_MAX, 0.008],
        color=FG,
        stroke_width=2.8,
    ).set_z_index(4)


def _water_fill(ax, level, n_samples=1200):
    xs = np.linspace(X_MIN, X_MAX, n_samples)
    ys = energy(xs)
    polys = []
    n = len(xs)

    def crossing(xa, ya, xb, yb, lvl):
        if yb == ya:
            return xa
        t = (lvl - ya) / (yb - ya)
        return xa + t * (xb - xa)

    i = 0
    while i < n - 1:
        if ys[i] >= level and ys[i + 1] >= level:
            i += 1
            continue
        run = []
        if ys[i] < level:
            run.append((float(xs[i]), float(ys[i])))
        else:
            xc = crossing(xs[i], ys[i], xs[i + 1], ys[i + 1], level)
            run.append((float(xc), float(level)))

        j = i
        while j < n - 1:
            if ys[j + 1] < level:
                run.append((float(xs[j + 1]), float(ys[j + 1])))
                j += 1
            else:
                xc = crossing(xs[j], ys[j], xs[j + 1], ys[j + 1], level)
                run.append((float(xc), float(level)))
                break

        pts = [ax.c2p(x, y) for (x, y) in run]
        pts.append(ax.c2p(run[-1][0], level))
        pts.append(ax.c2p(run[0][0], level))
        polys.append(Polygon(
            *pts, color=WATER_FILL, fill_color=WATER_FILL,
            fill_opacity=0.45, stroke_width=0,
        ))
        i = j + 1

    return VGroup(*polys).set_z_index(1)


def _water_surface(ax, level, n_samples=1200):
    xs = np.linspace(X_MIN, X_MAX, n_samples)
    ys = energy(xs)
    segments = []
    in_run = False
    run_start = 0.0
    for i in range(len(xs)):
        below = ys[i] < level
        if below and not in_run:
            if i == 0:
                run_start = float(xs[0])
            else:
                t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
                run_start = float(xs[i - 1] + t * (xs[i] - xs[i - 1]))
            in_run = True
        elif not below and in_run:
            t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            run_end = float(xs[i - 1] + t * (xs[i] - xs[i - 1]))
            segments.append((run_start, run_end))
            in_run = False
    if in_run:
        segments.append((run_start, float(xs[-1])))

    return VGroup(*[
        Line(
            ax.c2p(x1, level),
            ax.c2p(x2, level),
            color=WATER_LINE,
            stroke_width=3.0,
        )
        for (x1, x2) in segments
    ]).set_z_index(2)


# ===================================================================
# Metropolis-Hastings walker -- fixed proposal budget.
#
# The three panels must be directly comparable, so the walker is driven
# by a fixed number of proposals rather than a fixed number of accepts.
# A hop is accepted iff (i) the MH log-ratio passes, (ii) the chord
# between (x, E(x)) and (x_new, E(x_new)) stays strictly above the
# curve. The water-level weight 1/sqrt(level-E(x)) biases dwell toward
# high-potential-energy points, matching the physics used in
# glass_transition; an additional repulsion term encourages the chain
# to explore rather than loiter in one bin.
# ===================================================================

def _hop_valid(x0, x1, level, margin=0.0, min_dx=0.10, min_lift=0.04):
    if abs(x1 - x0) < min_dx:
        return False
    y0 = _energy_scalar(x0)
    y1 = _energy_scalar(x1)
    if y0 >= level or y1 >= level:
        return False
    lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
    mask = (_PROBE_XS > lo) & (_PROBE_XS < hi)
    xm = _PROBE_XS[mask]
    if xm.size == 0:
        return False
    t = (xm - x0) / (x1 - x0)
    y_line = y0 + t * (y1 - y0)
    curve_y = _PROBE_YS[mask]
    if np.any(y_line <= curve_y + margin):
        return False
    if float(np.max(y_line - curve_y)) < min_lift:
        return False
    return True


def _physics_weight(x, level):
    gap = level - _energy_scalar(x)
    if gap <= 1e-4:
        return 0.0
    return 1.0 / float(np.sqrt(gap))


def _walk_fixed_proposals(
    x0, sigma, level, n_proposals, seed,
    x_lo=None, x_hi=None,
    global_prob=0.0, repulsion=0.0, n_bins=80,
):
    """Run exactly n_proposals Metropolis-Hastings steps and return the
    accepted-x trajectory (an array starting at x0 and appending each
    accepted hop). Moves whose chord would touch or cross the curve are
    discarded after the MH accept check, so the returned trail always
    clears the landscape."""
    if x_lo is None:
        x_lo = X_MIN + 0.05
    if x_hi is None:
        x_hi = X_MAX - 0.05
    rng = np.random.default_rng(seed)
    x = float(x0)
    xs = [x]
    visits = np.zeros(n_bins, dtype=float)

    def bin_of(xv):
        f = (xv - x_lo) / (x_hi - x_lo)
        return int(np.clip(f * n_bins, 0, n_bins - 1))
    visits[bin_of(x)] += 1.0

    def log_weight(xv, vcount):
        pw = _physics_weight(xv, level)
        if pw <= 0.0:
            return -np.inf
        return float(np.log(pw) - repulsion * vcount)

    lw_curr = log_weight(x, visits[bin_of(x)])

    for _ in range(n_proposals):
        if global_prob > 0 and rng.random() < global_prob:
            x_new = rng.uniform(x_lo, x_hi)
        else:
            x_new = x + rng.normal(0.0, sigma)
        if x_new < x_lo or x_new > x_hi:
            continue
        vnew = visits[bin_of(x_new)]
        lw_new = log_weight(x_new, vnew)
        if not np.isfinite(lw_new):
            continue
        if lw_new < lw_curr and rng.random() > np.exp(lw_new - lw_curr):
            continue
        if not _hop_valid(x, x_new, level):
            continue
        x = x_new
        visits[bin_of(x)] += 1.0
        lw_curr = log_weight(x, visits[bin_of(x)])
        xs.append(x)
    return np.asarray(xs)


# ===================================================================
# Particle + trail at constant step_dt per accept.
#
# Each accepted hop takes exactly step_dt of wall-clock time and the
# particle advances linearly in (x, E(x)) across it. Panels with few
# accepts get a larger step_dt so all panels finish at the same wall
# clock -- the low-T particle moves slowly and deliberately, the
# high-T particle zips through, but neither stops and starts.
# ===================================================================

def _make_curve_particle(ax, xs_traj, step_dt):
    x0 = float(xs_traj[0])
    d = Dot(color=PARTICLE, radius=0.09).set_z_index(6)
    d.move_to(ax.c2p(x0, _energy_scalar(x0)))

    state = {"t": 0.0}
    n = len(xs_traj)
    total = max(0, n - 1) * step_dt

    def updater(mob, dt):
        state["t"] += dt
        if n < 2 or state["t"] >= total or step_dt <= 0:
            x = float(xs_traj[-1])
            y = _energy_scalar(x)
        else:
            t_scaled = state["t"] / step_dt
            idx = int(t_scaled)
            frac = t_scaled - idx
            xa = float(xs_traj[idx])
            xb = float(xs_traj[idx + 1])
            ya = _energy_scalar(xa)
            yb = _energy_scalar(xb)
            x = (1.0 - frac) * xa + frac * xb
            y = (1.0 - frac) * ya + frac * yb
        mob.move_to(ax.c2p(float(x), float(y)))

    return d, updater, total


def _make_segment_trail(ax, xs_traj, step_dt):
    pts = [ax.c2p(float(x), _energy_scalar(float(x))) for x in xs_traj]
    segments = [
        Line(pts[i], pts[i + 1], color=PARTICLE, stroke_width=3.0).set_z_index(3)
        for i in range(len(pts) - 1)
    ]
    for seg in segments:
        seg.set_stroke(opacity=0.0)
    container = VGroup(*segments)

    state = {"t": 0.0, "completed": 0}
    n_segs = len(segments)
    total = n_segs * step_dt

    def updater(mob, dt):
        state["t"] += dt
        if step_dt <= 0 or n_segs == 0:
            return
        t_elapsed = min(state["t"], total)
        idx = int(t_elapsed / step_dt)
        frac = (t_elapsed / step_dt) - idx

        while state["completed"] < idx and state["completed"] < n_segs:
            c = state["completed"]
            seg = segments[c]
            seg.set_stroke(opacity=1.0)
            seg.put_start_and_end_on(pts[c], pts[c + 1])
            state["completed"] = c + 1

        if idx < n_segs:
            seg = segments[idx]
            seg.set_stroke(opacity=1.0)
            start = np.array(pts[idx])
            end_full = np.array(pts[idx + 1])
            f = max(1e-3, min(1.0, frac))
            seg.put_start_and_end_on(start, start + (end_full - start) * f)

    return container, updater


# ===================================================================
# Simulation budget + panel configurations.
# ===================================================================

N_PROPOSALS = 700
X0 = 2.00           # start inside the deepest basin so the low-T chain is alive
SEED = 17
SIGMA = 0.55

PANEL_CONFIGS = (
    (PANEL_Y[0], BETA_HIGH, r"\beta_{\mathrm{high}}", r"$T \gg T_c$"),
    (PANEL_Y[1], BETA_CRIT, r"\beta_c",               r"$T \approx T_c$"),
    (PANEL_Y[2], BETA_LOW,  r"\beta_{\mathrm{low}}",  r"$T \ll T_c$"),
)


def _simulate_all():
    """Run the three chains once and return a list of (beta, xs)."""
    out = []
    for (_, beta, _, _) in PANEL_CONFIGS:
        level = level_from_beta(beta)
        xs = _walk_fixed_proposals(
            X0, sigma=SIGMA, level=level,
            n_proposals=N_PROPOSALS, seed=SEED,
            global_prob=0.25, repulsion=0.55,
        )
        out.append((beta, xs))
    return out


# ===================================================================
# Scene 1 -- three stacked energy panels, no title, centered vertically
# ===================================================================

class Scene1_ThreePanels(Scene):
    def construct(self):
        axes_list = []
        curves = []
        fills = []
        surfaces = []
        beta_lbls = []
        temp_lbls = []

        for (y_c, beta, beta_tex, temp_tex) in PANEL_CONFIGS:
            ax = _panel_axes(y_c)
            curve = _panel_curve(ax)
            level = level_from_beta(beta)
            wf = _water_fill(ax, level)
            ws = _water_surface(ax, level)
            bl = MathTex(beta_tex, color=FG, font_size=EQN_SIZE)
            bl.next_to(ax, LEFT, buff=0.30)
            tl = Tex(temp_tex, color=FG, font_size=LABEL_SIZE)
            tl.next_to(bl, DOWN, buff=0.16)

            axes_list.append(ax)
            curves.append(curve)
            fills.append(wf)
            surfaces.append(ws)
            beta_lbls.append(bl)
            temp_lbls.append(tl)

        self.play(
            *[FadeIn(ax) for ax in axes_list],
            *[FadeIn(bl) for bl in beta_lbls],
            *[FadeIn(tl) for tl in temp_lbls],
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[Create(c) for c in curves], lag_ratio=0.25),
            run_time=1.6,
        )
        self.play(
            *[FadeIn(wf) for wf in fills],
            *[FadeIn(ws) for ws in surfaces],
            run_time=0.8,
        )
        self.wait(0.4)

        chains = _simulate_all()
        WALL_CLOCK = 10.5

        particles = []
        p_updaters = []
        trails = []
        t_updaters = []
        totals = []

        for ax, (_, xs) in zip(axes_list, chains):
            n = len(xs)
            step_dt = WALL_CLOCK / max(1, n - 1)
            particle, p_up, total = _make_curve_particle(ax, xs, step_dt)
            trail, t_up = _make_segment_trail(ax, xs, step_dt)
            particles.append(particle)
            p_updaters.append(p_up)
            trails.append(trail)
            t_updaters.append(t_up)
            totals.append(total)

        for trail, particle in zip(trails, particles):
            self.add(trail, particle)
        for particle, p_up in zip(particles, p_updaters):
            particle.add_updater(p_up)
        for trail, t_up in zip(trails, t_updaters):
            trail.add_updater(t_up)

        self.wait(max(totals) + 0.15)

        for particle, p_up in zip(particles, p_updaters):
            particle.remove_updater(p_up)
        for trail, t_up in zip(trails, t_updaters):
            trail.remove_updater(t_up)

        self.wait(1.4)
