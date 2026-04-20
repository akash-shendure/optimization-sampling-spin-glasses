# visualizes the glass transition: liquid phase ergodic, glassy phase frozen in a basin
from pathlib import Path

import numpy as np
from manim import *

# manim cache lives next to the scene file so renders are scoped per-scene
_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".manim_cache"
config.media_dir = str(_CACHE)
config.video_dir = str(_HERE)
config.partial_movie_dir = str(_CACHE / "partial_movie_files")
config.quality = "high_quality"
config.frame_rate = 50
config.background_color = WHITE  # white bg for paper/print figures

# palette: black ink, blue "water" (thermal accessible region), red walker
FG = BLACK
WATER_FILL = BLUE_C
WATER_LINE = BLUE_E
PARTICLE = RED_E

EQN_SIZE = 42
CAPTION_SIZE = 36

# energy landscape domain and on-screen axis geometry
X_MIN, X_MAX = -5.0, 5.0
AXIS_WIDTH = 11.5
AXIS_HEIGHT = 5.2
AXIS_SHIFT = DOWN * 0.35  # leaves headroom for the beta slider up top

# smooth multi-well envelope — the macroscopic shape of E(x)
def _envelope(x):
    return (
        -1.75 * np.exp(-((x - 2.00) / 0.90) ** 2)
        - 1.30 * np.exp(-((x + 2.25) / 1.00) ** 2)
        - 0.95 * np.exp(-((x - 0.00) / 0.75) ** 2)
        - 0.80 * np.exp(-((x + 4.00) / 0.55) ** 2)
    )

# (freq, phase, amp) triples — incommensurate freqs produce rugged glassy bumps
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

# high-frequency ruggedness on top of the envelope — gives metastable wells
def _rugged(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * np.sin(f * x + p)
    return out

# full 1d toy energy: smooth wells + glassy roughness
def energy(x):
    x = np.asarray(x, dtype=float)
    return _envelope(x) + _rugged(x)

# scalar wrapper for manim plot/lambda callers
def _energy_scalar(x):
    return float(energy(np.array([float(x)]))[0])

# dense probe used for water level geometry and hop validation
_PROBE_XS = np.linspace(X_MIN, X_MAX, 6000)
_PROBE_YS = energy(_PROBE_XS)
_E_MIN = float(_PROBE_YS.min())
_E_MAX = float(_PROBE_YS.max())

Y_MIN = _E_MIN - 0.40
Y_MAX = _E_MAX + 0.80

# clamp the thermal "water line" so it never goes silly off-screen
E_FLOOR = _E_MIN - 0.05
LEVEL_CAP = _E_MAX + 0.65
THERMAL_C = 1.60  # cartoon: level ~ floor + C/beta (lower beta -> higher water)

# two regimes the scene contrasts
BETA_LIQUID = 0.35
BETA_GLASSY = 2.6

# map beta to a thermal "accessible" level above the energy floor
def level_from_beta(beta):
    return float(min(E_FLOOR + THERMAL_C / max(float(beta), 1e-3), LEVEL_CAP))

# x of the deepest point still below `level` — used to seed the glassy walker
def _deepest_below(level):
    ys_masked = np.where(_PROBE_YS < level, _PROBE_YS, np.inf)
    k = int(np.argmin(ys_masked))
    return float(_PROBE_XS[k])

# the contiguous accessible interval around x0 at a given water level (one basin)
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

# blank Axes — ticks suppressed since this is conceptual not quantitative
def _axes():
    return Axes(
        x_range=[X_MIN, X_MAX, 1],
        y_range=[Y_MIN, Y_MAX, 1],
        x_length=AXIS_WIDTH,
        y_length=AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(ORIGIN).shift(AXIS_SHIFT)

# the E(x) curve itself; tight dx for clean ruggedness, high z so it sits above water
def _curve(ax):
    return ax.plot(
        lambda x: _energy_scalar(x),
        x_range=[X_MIN, X_MAX, 0.008],
        color=FG,
        stroke_width=4.0,
    ).set_z_index(4)

# minimal E / x axis labels
def _axis_labels(ax):
    e_lbl = MathTex(r"E", color=FG, font_size=EQN_SIZE)
    e_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    x_lbl = MathTex(r"x", color=FG, font_size=EQN_SIZE)
    x_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(e_lbl, x_lbl)

# horizontal "water-line" segments wherever E(x) < level (visualizes accessible regions)
def _water_surface(ax, level, n_samples=1400):
    xs = np.linspace(X_MIN, X_MAX, n_samples)
    ys = energy(xs)
    segments = []
    in_run = False
    run_start = 0.0
    # linear-interp the crossings so the surface ends exactly on the curve
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

# filled polygons under the water line — visually flood the accessible basins
def _water_fill(ax, level, n_samples=1400):
    xs = np.linspace(X_MIN, X_MAX, n_samples)
    ys = energy(xs)
    polys = []
    n = len(xs)

    # interpolate the crossing x where the curve meets `level`
    def crossing(xa, ya, xb, yb, lvl):
        if yb == ya:
            return xa
        t = (lvl - ya) / (yb - ya)
        return xa + t * (xb - xa)

    # walk the curve, collecting each contiguous below-level "lake"
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

        # close the polygon along the level line back to the start
        pts = [ax.c2p(x, y) for (x, y) in run]
        pts.append(ax.c2p(run[-1][0], level))
        pts.append(ax.c2p(run[0][0], level))
        polys.append(Polygon(
            *pts, color=WATER_FILL, fill_color=WATER_FILL,
            fill_opacity=0.45, stroke_width=0,
        ))
        i = j + 1

    return VGroup(*polys).set_z_index(1)

# 1/sqrt(gap) — cartoon density-of-states near a level; weights time spent at depth
def _physics_weight(x, level):
    gap = level - _energy_scalar(x)
    if gap <= 1e-4:
        return 0.0
    return 1.0 / float(np.sqrt(gap))

# reject hops that pass through the curve — walker may only cross via water
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
    # the straight chord between endpoints must stay strictly above the curve
    t = (xm - x0) / (x1 - x0)
    y_line = y0 + t * (y1 - y0)
    curve_y = _PROBE_YS[mask]
    if np.any(y_line <= curve_y + margin):
        return False
    if float(np.max(y_line - curve_y)) < min_lift:
        return False
    return True

# cartoon mcmc walker — gaussian + occasional global jumps, with visit repulsion
# repulsion discourages re-visiting bins so animation looks like exploration
def _walk_hopping(
    x0, sigma, level, n_accepted, seed,
    x_lo=None, x_hi=None,
    global_prob=0.0, repulsion=0.0, n_bins=80,
    max_prop_per=500,
):
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

    # log weight = log(physics density) - repulsion * visits
    def log_weight(xv, vcount):
        pw = _physics_weight(xv, level)
        if pw <= 0.0:
            return -np.inf
        return float(np.log(pw) - repulsion * vcount)

    lw_curr = log_weight(x, visits[bin_of(x)])

    max_total = n_accepted * max_prop_per
    proposals = 0
    while len(xs) <= n_accepted and proposals < max_total:
        # mix local diffusion with rare global jumps (mimics swap/restart)
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
        # metropolis-style accept on the log weight ratio
        if lw_new < lw_curr and rng.random() > np.exp(lw_new - lw_curr):
            continue
        if not _hop_valid(x, x_new, level):
            continue
        x = x_new
        visits[bin_of(x)] += 1.0
        lw_curr = log_weight(x, visits[bin_of(x)])
        xs.append(x)
    return np.asarray(xs)

# dot riding the curve — linearly interpolates between accepted samples
def _make_curve_particle(ax, xs_traj, step_dt):
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

# segment-by-segment trail that pops in behind the particle as it moves
def _make_segment_trail(ax, xs_traj, step_dt):
    pts = [ax.c2p(float(x), _energy_scalar(float(x))) for x in xs_traj]
    segments = [
        Line(
            pts[i], pts[i + 1],
            color=PARTICLE, stroke_width=3.5,
        ).set_z_index(3)
        for i in range(len(pts) - 1)
    ]
    # start hidden — opacity is turned on segment-by-segment as time advances
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

        # snap all past segments fully visible
        while state["completed"] < idx and state["completed"] < n_segs:
            c = state["completed"]
            seg = segments[c]
            seg.set_stroke(opacity=1.0)
            seg.put_start_and_end_on(pts[c], pts[c + 1])
            state["completed"] = c + 1

        # draw the current segment partially, growing with frac
        if idx < n_segs:
            seg = segments[idx]
            seg.set_stroke(opacity=1.0)
            start = np.array(pts[idx])
            end_full = np.array(pts[idx + 1])
            f = max(1e-3, min(1.0, frac))
            seg.put_start_and_end_on(start, start + (end_full - start) * f)

    return container, updater

# starting level below the curve so the slider enters "empty" and fills up
_EMPTY_LEVEL = _E_MIN - 0.6

# little beta-slider widget at upper-right of frame
_SLIDER_CENTER = RIGHT * 4.55 + UP * 3.15
_SLIDER_LENGTH = 2.2

# slider showing beta low->high as water level drops from liquid -> glassy
def _beta_slider(level_tracker, liquid_level, glassy_level):
    left = _SLIDER_CENTER + LEFT * _SLIDER_LENGTH / 2
    right = _SLIDER_CENTER + RIGHT * _SLIDER_LENGTH / 2

    track = Line(left, right, color=FG, stroke_width=3).set_z_index(10)
    cap_left = Line(
        left + DOWN * 0.10, left + UP * 0.10, color=FG, stroke_width=3,
    ).set_z_index(10)
    cap_right = Line(
        right + DOWN * 0.10, right + UP * 0.10, color=FG, stroke_width=3,
    ).set_z_index(10)

    # beta label above, low/high anchors below
    title = MathTex(r"\beta", color=FG, font_size=38).next_to(
        track, UP, buff=0.20,
    ).set_z_index(10)
    lbl_low = Tex(r"low", color=FG, font_size=24).next_to(
        left, DOWN, buff=0.18,
    ).set_z_index(10)
    lbl_high = Tex(r"high", color=FG, font_size=24).next_to(
        right, DOWN, buff=0.18,
    ).set_z_index(10)

    # handle colored like the water — visually ties slider to fill level
    handle = Dot(color=WATER_LINE, radius=0.13).set_z_index(11)

    # invert: high beta -> low level -> handle to the right
    def place(m):
        lv = level_tracker.get_value()
        denom = liquid_level - glassy_level
        frac = 0.0 if denom == 0 else (liquid_level - lv) / denom
        frac = max(0.0, min(1.0, frac))
        m.move_to(left + RIGHT * _SLIDER_LENGTH * frac)

    handle.add_updater(place)
    return VGroup(track, cap_left, cap_right, title, lbl_low, lbl_high, handle)

# scene 1: thermal level rises and falls with beta to introduce the picture
class Scene1_ThermalLevel(Scene):

    def construct(self):
        # build the static landscape + slider
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        liquid_level = level_from_beta(BETA_LIQUID)
        glassy_level = level_from_beta(BETA_GLASSY)
        level_tracker = ValueTracker(_EMPTY_LEVEL)

        # always_redraw so fill/surface track the value tracker continuously
        water_fill = always_redraw(
            lambda: _water_fill(ax, level_tracker.get_value())
        )
        water_surface = always_redraw(
            lambda: _water_surface(ax, level_tracker.get_value())
        )
        slider = _beta_slider(level_tracker, liquid_level, glassy_level)

        # intro: axes -> curve -> slider, then attach water
        self.play(FadeIn(ax), Write(labels), run_time=0.8)
        self.play(Create(curve), run_time=2.0)
        self.play(FadeIn(slider), run_time=0.6)
        self.add(water_fill, water_surface)

        # raise water to the liquid level: ergodic, all basins connected
        self.play(
            level_tracker.animate.set_value(liquid_level),
            run_time=2.0, rate_func=smooth,
        )
        self.wait(1.0)
        # cool down: water drops, basins disconnect -> glassy regime
        self.play(
            level_tracker.animate.set_value(glassy_level),
            run_time=3.8, rate_func=smooth,
        )
        self.wait(1.2)
        # warm back up to close the loop
        self.play(
            level_tracker.animate.set_value(liquid_level),
            run_time=3.8, rate_func=smooth,
        )
        self.wait(1.3)

# scene 2: a single walker in the liquid phase exploring all basins freely
class Scene2_LiquidPhase(Scene):

    def construct(self):
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        # high-T water level covers (almost) the full landscape
        level = level_from_beta(BETA_LIQUID)
        water_fill = _water_fill(ax, level)
        water_surface = _water_surface(ax, level)

        self.add(ax, labels, curve, water_fill, water_surface)
        self.wait(0.5)

        caption = Tex(
            r"high temperature --- ``liquid'' phase",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        # walker starts far left and gets long mixing time + global jumps
        x0 = -3.8
        xs = _walk_hopping(
            x0, sigma=0.55, level=level, n_accepted=140, seed=11,
            global_prob=0.25, repulsion=1.0,
        )
        step_dt = 0.09  # per-sample dwell time on screen
        particle, p_updater, total = _make_curve_particle(ax, xs, step_dt=step_dt)
        trail, trail_updater = _make_segment_trail(ax, xs, step_dt=step_dt)

        # caption first, then attach walker, then let updaters run for `total`
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

# scene 3: start in liquid then cool below T_c -> walker freezes in one basin
class Scene3_GlassyPhase(Scene):

    def construct(self):
        ax = _axes()
        curve = _curve(ax)
        labels = _axis_labels(ax)

        liquid_level = level_from_beta(BETA_LIQUID)
        glassy_level = level_from_beta(BETA_GLASSY)

        # reuse a level tracker so water animates smoothly between phases
        level_tracker = ValueTracker(liquid_level)
        water_fill = always_redraw(
            lambda: _water_fill(ax, level_tracker.get_value())
        )
        water_surface = always_redraw(
            lambda: _water_surface(ax, level_tracker.get_value())
        )

        # opening state: show the same liquid-phase trace from scene 2 (frozen)
        caption_liquid = Tex(
            r"high temperature --- ``liquid'' phase",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        # reproduce scene 2's walk so the transition feels continuous
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

        # clear old walk to make room for the freeze sequence
        self.play(
            FadeOut(caption_liquid),
            FadeOut(liquid_trail),
            FadeOut(liquid_particle),
            run_time=0.9,
        )

        # cool down: water level drops below the inter-basin barriers
        self.play(
            level_tracker.animate.set_value(glassy_level),
            run_time=2.8, rate_func=smooth,
        )
        self.wait(0.3)

        caption_glassy = Tex(
            r"low temperature --- ``glassy'' phase (frozen)",
            color=FG, font_size=CAPTION_SIZE,
        ).to_edge(DOWN, buff=0.35)

        # seed inside the deepest connected basin and clip walker to its interval
        x_center = _deepest_below(glassy_level)
        x_lo, x_hi = _accessible_interval_around(x_center, glassy_level)
        x0 = x_lo + 0.30 * (x_hi - x_lo)
        # tighter sigma, no global jumps — walker is trapped in one basin
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

        # run the frozen walk
        particle.add_updater(p_updater)
        trail.add_updater(trail_updater)
        self.wait(total + 0.15)
        particle.remove_updater(p_updater)
        trail.remove_updater(trail_updater)
        self.wait(1.4)
