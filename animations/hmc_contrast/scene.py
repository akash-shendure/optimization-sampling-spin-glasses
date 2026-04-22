"""HMC vs Langevin contrast animations.

Two scenes on the same continuous-sampler comparison. Both balls start at
the same x and run with matched step counts; the difference is purely in
the dynamics (Euler--Maruyama on the gradient vs. leapfrog with periodic
momentum resamples).

Each scene is a two-panel layout:

    Top panel    -- the energy curve V(x) with two balls moving on it.
                    The ball motion alone tells you about instantaneous
                    speed (HMC accelerates down / decelerates up; Langevin
                    jitters).
    Bottom panel -- x(t) time trace. Here the qualitative difference
                    between the two samplers is unmissable: HMC draws a
                    smooth sinusoidal line, Langevin draws a noisy walk.

Scenes:

    Scene1_SmoothBasin      -- single-well quadratic basin. HMC sweeps
                               the full typical set in each leapfrog
                               integration; Langevin random-walks.
    Scene2_RuggedLandscape  -- the rugged glass-transition landscape,
                               started in a local minimum. Both balls
                               stay in the valley; HMC visibly climbs
                               toward the nearest barrier and reaches a
                               turning point short of the crest
                               (p^2/2 < Delta V).

Render with:

    manim scene.py Scene1_SmoothBasin
    manim scene.py Scene2_RuggedLandscape

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
LANGEVIN_COLOR = RED_E
HMC_COLOR = BLUE_E

EQN_SIZE = 38
LABEL_SIZE = 28
TRACE_LABEL_SIZE = 26


# ===================================================================
# Landscapes
# ===================================================================

def smooth_V(x):
    x = np.asarray(x, dtype=float)
    return 0.08 * x * x - 2.5


def smooth_dV(x):
    return 0.16 * float(x)


def smooth_V_scalar(x):
    return 0.08 * float(x) * float(x) - 2.5


# Rugged landscape -- duplicated from glass_transition/scene.py so the
# two subfolders stand alone.

_ENV_TERMS = (
    (-1.75, 2.00, 0.90),
    (-1.30, -2.25, 1.00),
    (-0.95, 0.00, 0.75),
    (-0.80, -4.00, 0.55),
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


def _envelope(x):
    out = np.zeros_like(x, dtype=float)
    for a, x0, w in _ENV_TERMS:
        out = out + a * np.exp(-((x - x0) / w) ** 2)
    return out


def _envelope_deriv(x):
    out = np.zeros_like(x, dtype=float)
    for a, x0, w in _ENV_TERMS:
        out = out + a * np.exp(-((x - x0) / w) ** 2) * (-2 * (x - x0) / (w * w))
    return out


def _rugged(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * np.sin(f * x + p)
    return out


def _rugged_deriv(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * f * np.cos(f * x + p)
    return out


def rugged_V(x):
    x = np.asarray(x, dtype=float)
    return _envelope(x) + _rugged(x)


def rugged_V_scalar(x):
    xa = np.array([float(x)])
    return float(rugged_V(xa)[0])


def rugged_dV(x):
    xa = np.array([float(x)])
    return float(_envelope_deriv(xa)[0] + _rugged_deriv(xa)[0])


# ===================================================================
# Two-panel layout
#   top panel    -- energy curve + moving balls
#   bottom panel -- x(t) time trace for both samplers
# ===================================================================

X_MIN, X_MAX = -5.0, 5.0
MAIN_AXIS_WIDTH = 11.8
MAIN_AXIS_HEIGHT = 4.4
MAIN_AXIS_POS = UP * 1.45

TRACE_AXIS_WIDTH = 11.8
TRACE_AXIS_HEIGHT = 2.1
TRACE_AXIS_POS = DOWN * 2.45

_RUG_PROBE_XS = np.linspace(X_MIN, X_MAX, 6000)
_RUG_PROBE_YS = rugged_V(_RUG_PROBE_XS)
RUG_Y_MIN = float(_RUG_PROBE_YS.min()) - 0.40
RUG_Y_MAX = float(_RUG_PROBE_YS.max()) + 0.60

SMOOTH_Y_MIN = -2.9
SMOOTH_Y_MAX = 0.4


def _main_axes(y_min, y_max):
    return Axes(
        x_range=[X_MIN, X_MAX, 1],
        y_range=[y_min, y_max, 1],
        x_length=MAIN_AXIS_WIDTH,
        y_length=MAIN_AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(MAIN_AXIS_POS)


def _trace_axes(n_steps, y_min=X_MIN, y_max=X_MAX):
    return Axes(
        x_range=[0, n_steps, n_steps],
        y_range=[y_min, y_max, 1],
        x_length=TRACE_AXIS_WIDTH,
        y_length=TRACE_AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(TRACE_AXIS_POS)


def _curve(ax, V_scalar):
    return ax.plot(
        lambda x: V_scalar(x),
        x_range=[X_MIN, X_MAX, 0.008],
        color=FG,
        stroke_width=4.0,
    ).set_z_index(4)


def _main_axis_labels(ax):
    v_lbl = MathTex(r"V(x)", color=FG, font_size=EQN_SIZE)
    v_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    x_lbl = MathTex(r"x", color=FG, font_size=EQN_SIZE)
    x_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(v_lbl, x_lbl)


def _trace_axis_labels(ax):
    x_lbl = MathTex(r"x(t)", color=FG, font_size=TRACE_LABEL_SIZE)
    x_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    t_lbl = Tex(r"time", color=FG, font_size=TRACE_LABEL_SIZE)
    t_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(x_lbl, t_lbl)


def _legend():
    hmc_dot = Dot(color=HMC_COLOR, radius=0.11)
    hmc_lbl = Tex(r"HMC", color=FG, font_size=LABEL_SIZE)
    hmc_row = VGroup(hmc_dot, hmc_lbl).arrange(RIGHT, buff=0.18)

    lan_dot = Dot(color=LANGEVIN_COLOR, radius=0.11)
    lan_lbl = Tex(r"Langevin", color=FG, font_size=LABEL_SIZE)
    lan_row = VGroup(lan_dot, lan_lbl).arrange(RIGHT, buff=0.18)

    legend = VGroup(hmc_row, lan_row).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    legend.to_corner(UR, buff=0.40).set_z_index(10)
    return legend


# ===================================================================
# Dynamics
# ===================================================================

def _simulate_leapfrog(
    x0, dV, n_refresh, L_steps, eps, beta, seed,
    x_lo=X_MIN + 0.05, x_hi=X_MAX - 0.05,
):
    """Standard HMC: momentum ~ N(0, 1/sqrt(beta)) at the start of each
    integration, L leapfrog steps with step size eps, no Metropolis
    correction (visually identical at this scale). Reflected at plot
    boundaries so the ball stays on screen."""
    rng = np.random.default_rng(seed)
    x = float(x0)
    sigma_p = 1.0 / float(np.sqrt(beta))
    xs = [x]
    for _ in range(n_refresh):
        p = float(rng.normal(0.0, sigma_p))
        for _ in range(L_steps):
            p = p - 0.5 * eps * dV(x)
            x = x + eps * p
            if x < x_lo:
                x = x_lo + (x_lo - x)
                p = -p
            elif x > x_hi:
                x = x_hi - (x - x_hi)
                p = -p
            p = p - 0.5 * eps * dV(x)
            xs.append(x)
    return np.asarray(xs)


def _simulate_langevin(
    x0, dV, n_steps, eps, beta, seed,
    x_lo=X_MIN + 0.05, x_hi=X_MAX - 0.05,
):
    """Overdamped Langevin (Euler-Maruyama). Stationary distribution is
    exp(-beta V)/Z. Reflected at plot boundaries."""
    rng = np.random.default_rng(seed)
    x = float(x0)
    noise_scale = float(np.sqrt(2.0 * eps / beta))
    xs = [x]
    for _ in range(n_steps):
        x = x - eps * dV(x) + noise_scale * float(rng.normal())
        if x < x_lo:
            x = x_lo + (x_lo - x)
        elif x > x_hi:
            x = x_hi - (x - x_hi)
        xs.append(x)
    return np.asarray(xs)


def _x_at_time(xs_traj, step_dt, t):
    """Linear interpolation of xs_traj at time t. Clamps to the last
    sample once t exceeds the trajectory's total duration."""
    n = len(xs_traj)
    total = (n - 1) * step_dt
    if t >= total:
        return float(xs_traj[-1])
    if t <= 0.0:
        return float(xs_traj[0])
    t_scaled = t / step_dt
    idx = int(t_scaled)
    frac = t_scaled - idx
    return (1.0 - frac) * float(xs_traj[idx]) + frac * float(xs_traj[idx + 1])


def _make_ball_updater(main_ax, V_scalar, xs_traj, step_dt, t_tracker):
    """Updater that places a dot on the curve y = V(x) at position
    x = xs_traj(t)."""
    def _upd(mob, dt):
        t = t_tracker.get_value()
        x = _x_at_time(xs_traj, step_dt, t)
        y = V_scalar(x)
        mob.move_to(main_ax.c2p(x, y))
    return _upd


def _make_trace_helper_updater(trace_ax, xs_traj, step_dt, t_tracker):
    """Updater that places an invisible helper dot at screen-space
    coordinates (step_index, x(step_index)). TracedPath attached to this
    helper dot draws the growing x(t) line."""
    def _upd(mob, dt):
        t = t_tracker.get_value()
        x = _x_at_time(xs_traj, step_dt, t)
        step = t / step_dt
        mob.move_to(trace_ax.c2p(step, x))
    return _upd


# ===================================================================
# Scenes
# ===================================================================

def _run_two_panel_scene(
    scene, V_scalar, y_min, y_max, xs_hmc, xs_lan, step_dt,
    trace_y_min=None, trace_y_max=None,
):
    """Shared playback routine. Both scenes differ only in the landscape,
    the trajectories, and (optionally) the zoom on the trace panel."""
    n_matched = len(xs_hmc) - 1
    total = n_matched * step_dt

    main_ax = _main_axes(y_min, y_max)
    curve = _curve(main_ax, V_scalar)
    main_lbls = _main_axis_labels(main_ax)

    ty_min = X_MIN if trace_y_min is None else trace_y_min
    ty_max = X_MAX if trace_y_max is None else trace_y_max
    trace_ax = _trace_axes(n_matched, ty_min, ty_max)
    trace_lbls = _trace_axis_labels(trace_ax)

    legend = _legend()

    t_tracker = ValueTracker(0.0)

    # Main-panel balls (on the energy curve).
    p_hmc = Dot(color=HMC_COLOR, radius=0.13).set_z_index(7)
    p_hmc.move_to(main_ax.c2p(float(xs_hmc[0]), V_scalar(float(xs_hmc[0]))))
    p_hmc.add_updater(_make_ball_updater(main_ax, V_scalar, xs_hmc, step_dt, t_tracker))

    p_lan = Dot(color=LANGEVIN_COLOR, radius=0.13).set_z_index(7)
    p_lan.move_to(main_ax.c2p(float(xs_lan[0]), V_scalar(float(xs_lan[0]))))
    p_lan.add_updater(_make_ball_updater(main_ax, V_scalar, xs_lan, step_dt, t_tracker))

    # Invisible helper dots that ride the trace panel, with TracedPath
    # behind them drawing the growing x(t) line.
    helper_hmc = Dot(radius=0.001, color=HMC_COLOR).set_opacity(0.0)
    helper_hmc.move_to(trace_ax.c2p(0.0, float(xs_hmc[0])))
    helper_hmc.add_updater(_make_trace_helper_updater(trace_ax, xs_hmc, step_dt, t_tracker))

    helper_lan = Dot(radius=0.001, color=LANGEVIN_COLOR).set_opacity(0.0)
    helper_lan.move_to(trace_ax.c2p(0.0, float(xs_lan[0])))
    helper_lan.add_updater(_make_trace_helper_updater(trace_ax, xs_lan, step_dt, t_tracker))

    trace_line_hmc = TracedPath(
        helper_hmc.get_center, stroke_color=HMC_COLOR, stroke_width=3.0,
    ).set_z_index(3)
    trace_line_lan = TracedPath(
        helper_lan.get_center, stroke_color=LANGEVIN_COLOR, stroke_width=3.0,
    ).set_z_index(3)

    # --- playback ---
    scene.play(FadeIn(main_ax), Write(main_lbls), run_time=0.6)
    scene.play(Create(curve), run_time=1.1)
    scene.play(FadeIn(trace_ax), Write(trace_lbls), FadeIn(legend), run_time=0.5)

    scene.add(trace_line_hmc, trace_line_lan, helper_hmc, helper_lan, p_hmc, p_lan)
    scene.wait(0.2)

    scene.play(
        t_tracker.animate.set_value(total),
        run_time=total, rate_func=linear,
    )

    p_hmc.clear_updaters()
    p_lan.clear_updaters()
    helper_hmc.clear_updaters()
    helper_lan.clear_updaters()
    scene.wait(1.0)


class Scene1_SmoothBasin(Scene):
    """Unimodal smooth basin. HMC sweeps the full typical set in each
    leapfrog integration; Langevin random-walks from the same start."""

    def construct(self):
        beta = 1.0
        x_start = -3.6
        eps_hmc = 0.20
        L_steps = 80
        n_refresh = 3
        xs_hmc = _simulate_leapfrog(
            x0=x_start, dV=smooth_dV, n_refresh=n_refresh, L_steps=L_steps,
            eps=eps_hmc, beta=beta, seed=13,
        )
        n_matched = len(xs_hmc) - 1
        eps_lan = 0.03
        xs_lan = _simulate_langevin(
            x0=x_start, dV=smooth_dV, n_steps=n_matched,
            eps=eps_lan, beta=beta, seed=19,
        )

        _run_two_panel_scene(
            self, smooth_V_scalar, SMOOTH_Y_MIN, SMOOTH_Y_MAX,
            xs_hmc, xs_lan, step_dt=0.020,
        )


class Scene2_RuggedLandscape(Scene):
    """Rugged glass-transition landscape, started on the left at the
    x = -2.25 envelope well. Langevin (seed 60) accumulates enough
    stochastic noise to cross the big envelope barrier near x = 0 at
    about step 126 and settle into the right-side global minimum near
    x = 1.777. HMC (seed 8) at the same beta has kinetic energy
    p^2 / 2 < Delta V for every momentum resample, so its leapfrog
    trajectories stay confined to the two left envelope wells (x in
    [-4.4, -1.3]) and never cross to x > 0."""

    def construct(self):
        beta = 2.0
        x_start = -2.25
        eps_hmc = 0.06
        L_steps = 120
        n_refresh = 4
        xs_hmc = _simulate_leapfrog(
            x0=x_start, dV=rugged_dV, n_refresh=n_refresh, L_steps=L_steps,
            eps=eps_hmc, beta=beta, seed=8,
        )
        n_matched = len(xs_hmc) - 1
        eps_lan = 0.05
        xs_lan = _simulate_langevin(
            x0=x_start, dV=rugged_dV, n_steps=n_matched,
            eps=eps_lan, beta=beta, seed=60,
        )

        _run_two_panel_scene(
            self, rugged_V_scalar, RUG_Y_MIN, RUG_Y_MAX,
            xs_hmc, xs_lan, step_dt=0.013,
        )
