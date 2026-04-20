"""Glass-transition animations.

Three scenes that visualize how a particle searches a rugged 1D energy
landscape as temperature changes:

    Scene1_ThermalLevel — plot E(x) and fade the water level between
        the liquid (high) and glassy (low) levels: liquid -> glassy ->
        liquid.
    Scene2_LiquidPhase  — the water line sits above every peak, so the
        particle hops freely across the landscape. New x values are
        sampled from p(x) = 1/sqrt(level - E(x)) (classical dwell-time
        density), biasing proposals toward high-potential-energy points.
    Scene3_GlassyPhase  — water is just above the deepest basin; the
        same sampler biases proposals toward the turning points at the
        basin's edges, and the particle is trapped in one pool.

A proposed hop from (x, E(x)) to (x_new, E(x_new)) is accepted iff the
straight line between those two points stays above the energy curve.
Invalid proposals are silently discarded; only accepted hops are shown,
each taking one fixed time step.

Render with:

    manim scene.py Scene1_ThermalLevel
    manim scene.py Scene2_LiquidPhase
    manim scene.py Scene3_GlassyPhase

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

EQN_SIZE = 42
CAPTION_SIZE = 36

# ---------- x range ----------
X_MIN, X_MAX = -5.0, 5.0
AXIS_WIDTH = 11.5
AXIS_HEIGHT = 5.2
AXIS_SHIFT = DOWN * 0.35


# ---------- densely rugged smooth landscape ----------
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


# Probe the curve once to derive visual / thermal constants.
_PROBE_XS = np.linspace(X_MIN, X_MAX, 6000)
_PROBE_YS = energy(_PROBE_XS)
_E_MIN = float(_PROBE_YS.min())
_E_MAX = float(_PROBE_YS.max())

Y_MIN = _E_MIN - 0.40
Y_MAX = _E_MAX + 0.80

E_FLOOR = _E_MIN - 0.05
LEVEL_CAP = _E_MAX + 0.65
THERMAL_C = 1.60

BETA_LIQUID = 0.35
BETA_GLASSY = 2.6


def level_from_beta(beta):
    return float(min(E_FLOOR + THERMAL_C / max(float(beta), 1e-3), LEVEL_CAP))


# ===================================================================
# Accessible-region helpers
# ===================================================================

def _deepest_below(level):
    ys_masked = np.where(_PROBE_YS < level, _PROBE_YS, np.inf)
    k = int(np.argmin(ys_masked))
    return float(_PROBE_XS[k])


def _accessible_interval_around(x0, level):
    k = int(np.argmin(np.abs(_PROBE_XS - x0)))
    xs, ys = _PROBE_XS, _PROBE_YS
    if ys[k] >= level:
        return x0, x0
    lo = k
    while lo > 0 and ys[lo - 1] < level:
        lo -= 1
    hi = k
    while hi < len(xs) - 1 and ys[hi + 1] < level:
        hi += 1
    return float(xs[lo]), float(xs[hi])


# ===================================================================
# Scene-building helpers
# ===================================================================

def _axes():
    return Axes(
        x_range=[X_MIN, X_MAX, 1],
        y_range=[Y_MIN, Y_MAX, 1],
        x_length=AXIS_WIDTH,
        y_length=AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(ORIGIN).shift(AXIS_SHIFT)


def _curve(ax):
    return ax.plot(
        lambda x: _energy_scalar(x),
        x_range=[X_MIN, X_MAX, 0.008],
        color=FG,
        stroke_width=4.0,
    ).set_z_index(4)


def _axis_labels(ax):
    e_lbl = MathTex(r"E", color=FG, font_size=EQN_SIZE)
    e_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    x_lbl = MathTex(r"x", color=FG, font_size=EQN_SIZE)
    x_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(e_lbl, x_lbl)


def _water_surface(ax, level, n_samples=1400):
    """Horizontal surface segments only over x-intervals where
    E(x) < level. Each contiguous pool gets one short line; the
    surface disappears where land (the energy curve) sticks out."""
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
            stroke_width=4.5,
        )
        for (x1, x2) in segments
    ]).set_z_index(2)


def _water_fill(ax, level, n_samples=1400):
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


# ===================================================================
# Physics-weighted stationary distribution p(x) ∝ 1 / sqrt(level - E(x))
#
# For a classical particle bouncing inside a potential well with total
# energy = level, the time-average probability density at position x is
# dt/dx = 1/v(x) ∝ 1/sqrt(level - V(x)). Near classical turning points
# (where E(x) approaches level) the density diverges (integrable) —
# the particle is moving slowly there and spends more of its time at
# those high-PE positions. We reproduce this distribution via
# Metropolis-Hastings on a local Gaussian proposal: the walk stays
# near the current x (so each hop clears the strict no-crossing check)
# but preferentially dwells at high-PE points.
# ===================================================================

def _physics_weight(x, level):
    gap = level - _energy_scalar(x)
    if gap <= 1e-4:
        return 0.0
    return 1.0 / float(np.sqrt(gap))


# ===================================================================
# Hopping walk
# ===================================================================

def _hop_valid(x0, x1, level, margin=0.0, min_dx=0.10, min_lift=0.04):
    """A hop from (x0, E(x0)) to (x1, E(x1)) is valid iff:
      1. The endpoints are at least `min_dx` apart. Very short hops
         track the curve's local tangent too closely — visually the
         line just slides along the curve and never lifts off.
      2. Both endpoints lie below `level`.
      3. The straight line between them stays strictly above the
         curve everywhere in between.
      4. The maximum gap between the line and the curve (measured
         over the interior) exceeds `min_lift`, i.e. the line
         visibly lifts off the landscape rather than running
         tangent to it.
    The check uses the 6000-point probe grid."""
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


def _walk_hopping(
    x0, sigma, level, n_accepted, seed,
    x_lo=None, x_hi=None,
    global_prob=0.0, repulsion=0.0, n_bins=80,
    max_prop_per=500,
):
    """Metropolis-Hastings random walk on x with a self-avoiding bias.

    Proposal: x + N(0, σ²) with probability (1 - global_prob), or a
    uniform draw on [x_lo, x_hi] with probability global_prob.

    Acceptance uses an effective weight
        W(x) = (1/sqrt(level - E(x))) * exp(-repulsion * visits[bin(x)])
    that combines the classical dwell-time density (biasing toward
    high-PE points) with a repulsion term that discourages returning
    to bins the walker has already visited. Both proposal distributions
    are symmetric, so the M-H ratio is simply W(x_new)/W(x_curr).

    After each acceptance, the visit count for the new bin is
    incremented, so the acceptance weights drift over time. Finally,
    the strict hop-validity test rejects any move whose straight line
    would touch or cross the curve.

    Rejected proposals are discarded silently; only accepted x values
    are returned."""
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

    max_total = n_accepted * max_prop_per
    proposals = 0
    while len(xs) <= n_accepted and proposals < max_total:
        if global_prob > 0 and rng.random() < global_prob:
            x_new = rng.uniform(x_lo, x_hi)
        else:
            x_new = x + rng.normal(0.0, sigma)
        proposals += 1
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


def _make_curve_particle(ax, xs_traj, step_dt):
    """Particle interpolates linearly in (x, y=E) between consecutive
    accepted x values, so the traced path draws each accepted hop as a
    straight line segment in the (x, E) plane."""
    x0 = float(xs_traj[0])
    d = Dot(color=PARTICLE, radius=0.12).set_z_index(6)
    d.move_to(ax.c2p(x0, _energy_scalar(x0)))
    state = {"t": 0.0}
    n = len(xs_traj)
    total = (n - 1) * step_dt

    def updater(mob, dt):
        state["t"] += dt
        if state["t"] >= total:
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
    """Build the trace as a list of distinct Line segments — one per
    accepted hop between (x_i, E(x_i)) and (x_{i+1}, E(x_{i+1})). All
    segments are preconstructed at full length but initially hidden,
    and the returned updater progressively reveals them in sync with
    the particle: the current hop's segment grows from its starting
    endpoint out to the particle's live position; completed hops'
    segments are locked in at full length with exact endpoints on
    the curve."""
    pts = [ax.c2p(float(x), _energy_scalar(float(x))) for x in xs_traj]
    segments = [
        Line(
            pts[i], pts[i + 1],
            color=PARTICLE, stroke_width=3.5,
        ).set_z_index(3)
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
        t_elapsed = min(state["t"], total)
        idx = int(t_elapsed / step_dt)
        frac = (t_elapsed / step_dt) - idx

        # Lock in any newly-completed segments at full length.
        while state["completed"] < idx and state["completed"] < n_segs:
            c = state["completed"]
            seg = segments[c]
            seg.set_stroke(opacity=1.0)
            seg.put_start_and_end_on(pts[c], pts[c + 1])
            state["completed"] = c + 1

        # Grow the in-progress segment to match the particle's current x.
        if idx < n_segs:
            seg = segments[idx]
            seg.set_stroke(opacity=1.0)
            start = np.array(pts[idx])
            end_full = np.array(pts[idx + 1])
            f = max(1e-3, min(1.0, frac))
            seg.put_start_and_end_on(start, start + (end_full - start) * f)

    return container, updater


# ===================================================================
# Scenes
# ===================================================================

_EMPTY_LEVEL = _E_MIN - 0.6  # below every point on the curve → empty water

# ---- beta slider (Scene 1) ----
_SLIDER_CENTER = RIGHT * 4.55 + UP * 3.15
_SLIDER_LENGTH = 2.2


def _beta_slider(level_tracker, liquid_level, glassy_level):
    """Horizontal slider in the top-right that tracks the water level.
    Handle sits at the left when the surface is at the liquid level
    (small β) and at the right when the surface is at the glassy level
    (large β), interpolated linearly in between."""
    left = _SLIDER_CENTER + LEFT * _SLIDER_LENGTH / 2
    right = _SLIDER_CENTER + RIGHT * _SLIDER_LENGTH / 2

    track = Line(left, right, color=FG, stroke_width=3).set_z_index(10)
    cap_left = Line(
        left + DOWN * 0.10, left + UP * 0.10, color=FG, stroke_width=3,
    ).set_z_index(10)
    cap_right = Line(
        right + DOWN * 0.10, right + UP * 0.10, color=FG, stroke_width=3,
    ).set_z_index(10)

    title = MathTex(r"\beta", color=FG, font_size=38).next_to(
        track, UP, buff=0.20,
    ).set_z_index(10)
    lbl_low = Tex(r"low", color=FG, font_size=24).next_to(
        left, DOWN, buff=0.18,
    ).set_z_index(10)
    lbl_high = Tex(r"high", color=FG, font_size=24).next_to(
        right, DOWN, buff=0.18,
    ).set_z_index(10)

    handle = Dot(color=WATER_LINE, radius=0.13).set_z_index(11)

    def place(m):
        lv = level_tracker.get_value()
        denom = liquid_level - glassy_level
        frac = 0.0 if denom == 0 else (liquid_level - lv) / denom
        frac = max(0.0, min(1.0, frac))
        m.move_to(left + RIGHT * _SLIDER_LENGTH * frac)

    handle.add_updater(place)
    return VGroup(track, cap_left, cap_right, title, lbl_low, lbl_high, handle)


class Scene1_ThermalLevel(Scene):
    """Water rises from empty to the liquid level, sweeps down to the
    glassy level, and back up to liquid. The β slider in the top right
    tracks the motion of the surface."""

    def construct(self):
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        liquid_level = level_from_beta(BETA_LIQUID)
        glassy_level = level_from_beta(BETA_GLASSY)
        level_tracker = ValueTracker(_EMPTY_LEVEL)

        water_fill = always_redraw(
            lambda: _water_fill(ax, level_tracker.get_value())
        )
        water_surface = always_redraw(
            lambda: _water_surface(ax, level_tracker.get_value())
        )
        slider = _beta_slider(level_tracker, liquid_level, glassy_level)

        self.play(FadeIn(ax), Write(labels), run_time=0.8)
        self.play(Create(curve), run_time=2.0)
        self.play(FadeIn(slider), run_time=0.6)
        self.add(water_fill, water_surface)

        self.play(
            level_tracker.animate.set_value(liquid_level),
            run_time=2.0, rate_func=smooth,
        )
        self.wait(1.0)
        self.play(
            level_tracker.animate.set_value(glassy_level),
            run_time=3.8, rate_func=smooth,
        )
        self.wait(1.2)
        self.play(
            level_tracker.animate.set_value(liquid_level),
            run_time=3.8, rate_func=smooth,
        )
        self.wait(1.3)


class Scene2_LiquidPhase(Scene):
    """High temperature. Opens with the static liquid-level water
    inherited from Scene 1 and closes with the caption still visible
    so Scene 3 can inherit it and fade it out cleanly."""

    def construct(self):
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        level = level_from_beta(BETA_LIQUID)
        water_fill = _water_fill(ax, level)
        water_surface = _water_surface(ax, level)

        self.add(ax, labels, curve, water_fill, water_surface)
        self.wait(0.5)

        caption = Tex(
            r"high temperature --- ``liquid'' phase",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        x0 = -3.8
        xs = _walk_hopping(
            x0, sigma=0.55, level=level, n_accepted=140, seed=11,
            global_prob=0.25, repulsion=1.0,
        )
        step_dt = 0.09
        particle, p_updater, total = _make_curve_particle(ax, xs, step_dt=step_dt)
        trail, trail_updater = _make_segment_trail(ax, xs, step_dt=step_dt)

        self.play(FadeIn(caption), run_time=0.5)
        self.wait(0.2)
        self.add(trail, particle)
        self.wait(0.3)

        particle.add_updater(p_updater)
        trail.add_updater(trail_updater)
        self.wait(total + 0.15)
        particle.remove_updater(p_updater)
        trail.remove_updater(trail_updater)
        self.wait(1.5)
        # Scene 2 ends with the caption, trail, and particle all still
        # visible. Scene 3 inherits that frame and fades them out.


class Scene3_GlassyPhase(Scene):
    """Low temperature. Opens with the liquid-water baseline + the
    Scene 2 caption, fades that caption out, animates the surface down
    to the glassy level, then runs the confined walk."""

    def construct(self):
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        liquid_level = level_from_beta(BETA_LIQUID)
        glassy_level = level_from_beta(BETA_GLASSY)

        level_tracker = ValueTracker(liquid_level)
        water_fill = always_redraw(
            lambda: _water_fill(ax, level_tracker.get_value())
        )
        water_surface = always_redraw(
            lambda: _water_surface(ax, level_tracker.get_value())
        )

        # Reproduce Scene 2's final frame: caption + full trail + particle
        # at the last walked x, all over the liquid-level water.
        caption_liquid = Tex(
            r"high temperature --- ``liquid'' phase",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        liquid_xs = _walk_hopping(
            -3.8, sigma=0.55, level=liquid_level, n_accepted=140, seed=11,
            global_prob=0.25, repulsion=1.0,
        )
        liquid_pts = [
            ax.c2p(float(x), _energy_scalar(float(x))) for x in liquid_xs
        ]
        liquid_trail = VGroup(*[
            Line(
                liquid_pts[i], liquid_pts[i + 1],
                color=PARTICLE, stroke_width=3.5,
            ).set_z_index(3)
            for i in range(len(liquid_pts) - 1)
        ])
        liquid_particle = Dot(
            color=PARTICLE, radius=0.12,
        ).move_to(liquid_pts[-1]).set_z_index(6)

        self.add(
            ax, labels, curve, water_fill, water_surface,
            caption_liquid, liquid_trail, liquid_particle,
        )
        self.wait(0.6)

        # Fade out everything Scene 2 left behind before the surface drops.
        self.play(
            FadeOut(caption_liquid),
            FadeOut(liquid_trail),
            FadeOut(liquid_particle),
            run_time=0.9,
        )

        # Surface drops from liquid to glassy; pools shrink to one puddle.
        self.play(
            level_tracker.animate.set_value(glassy_level),
            run_time=2.8, rate_func=smooth,
        )
        self.wait(0.3)

        caption_glassy = Tex(
            r"low temperature --- ``glassy'' phase (frozen)",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        x_center = _deepest_below(glassy_level)
        x_lo, x_hi = _accessible_interval_around(x_center, glassy_level)
        x0 = x_lo + 0.30 * (x_hi - x_lo)
        xs = _walk_hopping(
            x0, sigma=0.22, level=glassy_level, n_accepted=80, seed=23,
            x_lo=x_lo, x_hi=x_hi, repulsion=0.8,
        )
        step_dt = 0.11
        particle, p_updater, total = _make_curve_particle(ax, xs, step_dt=step_dt)
        trail, trail_updater = _make_segment_trail(ax, xs, step_dt=step_dt)

        self.play(FadeIn(caption_glassy), run_time=0.5)
        self.wait(0.2)
        self.add(trail, particle)
        self.wait(0.3)

        particle.add_updater(p_updater)
        trail.add_updater(trail_updater)
        self.wait(total + 0.15)
        particle.remove_updater(p_updater)
        trail.remove_updater(trail_updater)
        self.wait(1.4)
