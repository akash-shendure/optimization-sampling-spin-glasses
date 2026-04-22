# HMC vs Langevin trajectories on the same potential — HMC glides ballistically, Langevin diffuses
from pathlib import Path

import numpy as np
from manim import *

# render config — local cache, white slide background
_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".manim_cache"
config.media_dir = str(_CACHE)
config.video_dir = str(_HERE)
config.partial_movie_dir = str(_CACHE / "partial_movie_files")
config.quality = "high_quality"
config.frame_rate = 50
config.background_color = WHITE

# palette: red for Langevin (random-walk), blue for HMC (momentum-aided)
FG = BLACK
LANGEVIN_COLOR = RED_E
HMC_COLOR = BLUE_E

EQN_SIZE = 38
LABEL_SIZE = 28
TRACE_LABEL_SIZE = 26

# smooth quadratic potential — convex bowl with minimum near 0
def smooth_V(x):
    x = np.asarray(x, dtype=float)
    return 0.08 * x * x - 2.5

# derivative — needed by both leapfrog and Langevin updates
def smooth_dV(x):
    return 0.16 * float(x)

# scalar version for manim's plot()
def smooth_V_scalar(x):
    return 0.08 * float(x) * float(x) - 2.5

# (amp, center, width) of envelope gaussians — basin skeleton for the rugged potential
_ENV_TERMS = (
    (-1.75, 2.00, 0.90),
    (-1.30, -2.25, 1.00),
    (-0.95, 0.00, 0.75),
    (-0.80, -4.00, 0.55),
)

# (freq, phase, amp) for rugged ripples — same modes as the arrhenius scene for a familiar look
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

# envelope = sum of gaussian wells
def _envelope(x):
    out = np.zeros_like(x, dtype=float)
    for a, x0, w in _ENV_TERMS:
        out = out + a * np.exp(-((x - x0) / w) ** 2)
    return out

# analytic derivative of the envelope — chain rule on each gaussian
def _envelope_deriv(x):
    out = np.zeros_like(x, dtype=float)
    for a, x0, w in _ENV_TERMS:
        out = out + a * np.exp(-((x - x0) / w) ** 2) * (-2 * (x - x0) / (w * w))
    return out

# sum of sinusoidal modes layered on the envelope
def _rugged(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * np.sin(f * x + p)
    return out

# d/dx of the rugged piece
def _rugged_deriv(x):
    out = np.zeros_like(x, dtype=float)
    for f, p, a in _MODES:
        out = out + a * f * np.cos(f * x + p)
    return out

# full rugged potential used in scene 2
def rugged_V(x):
    x = np.asarray(x, dtype=float)
    return _envelope(x) + _rugged(x)

# scalar wrapper for plot()
def rugged_V_scalar(x):
    xa = np.array([float(x)])
    return float(rugged_V(xa)[0])

# scalar gradient — used by both dynamics for the rugged scene
def rugged_dV(x):
    xa = np.array([float(x)])
    return float(_envelope_deriv(xa)[0] + _rugged_deriv(xa)[0])

# main potential plot lives in the upper half, trace plot in the lower half
X_MIN, X_MAX = -5.0, 5.0
MAIN_AXIS_WIDTH = 11.8
MAIN_AXIS_HEIGHT = 4.4
MAIN_AXIS_POS = UP * 1.45

TRACE_AXIS_WIDTH = 11.8
TRACE_AXIS_HEIGHT = 2.1
TRACE_AXIS_POS = DOWN * 2.45

# precompute rugged y-range from a dense probe
_RUG_PROBE_XS = np.linspace(X_MIN, X_MAX, 6000)
_RUG_PROBE_YS = rugged_V(_RUG_PROBE_XS)
RUG_Y_MIN = float(_RUG_PROBE_YS.min()) - 0.40
RUG_Y_MAX = float(_RUG_PROBE_YS.max()) + 0.60

# fixed y-range for the smooth bowl — chosen to match smooth_V's image roughly
SMOOTH_Y_MIN = -2.9
SMOOTH_Y_MAX = 0.4

# top panel: V(x) curve
def _main_axes(y_min, y_max):
    return Axes(
        x_range=[X_MIN, X_MAX, 1],
        y_range=[y_min, y_max, 1],
        x_length=MAIN_AXIS_WIDTH,
        y_length=MAIN_AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(MAIN_AXIS_POS)

# bottom panel: x(t) trace — x range is "step count" so single tick at the end
def _trace_axes(n_steps, y_min=X_MIN, y_max=X_MAX):
    return Axes(
        x_range=[0, n_steps, n_steps],
        y_range=[y_min, y_max, 1],
        x_length=TRACE_AXIS_WIDTH,
        y_length=TRACE_AXIS_HEIGHT,
        tips=False,
        axis_config={"color": FG, "stroke_width": 2.2, "include_ticks": False},
    ).move_to(TRACE_AXIS_POS)

# plotted potential curve — z=4 so balls and trails draw underneath
def _curve(ax, V_scalar):
    return ax.plot(
        lambda x: V_scalar(x),
        x_range=[X_MIN, X_MAX, 0.008],
        color=FG,
        stroke_width=4.0,
    ).set_z_index(4)

# V(x) and x labels on the main axes
def _main_axis_labels(ax):
    v_lbl = MathTex(r"V(x)", color=FG, font_size=EQN_SIZE)
    v_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    x_lbl = MathTex(r"x", color=FG, font_size=EQN_SIZE)
    x_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(v_lbl, x_lbl)

# x(t) and time labels on the trace axes
def _trace_axis_labels(ax):
    x_lbl = MathTex(r"x(t)", color=FG, font_size=TRACE_LABEL_SIZE)
    x_lbl.next_to(ax.y_axis.get_top(), LEFT, buff=0.15)
    t_lbl = Tex(r"time", color=FG, font_size=TRACE_LABEL_SIZE)
    t_lbl.next_to(ax.x_axis.get_right(), DOWN, buff=0.20)
    return VGroup(x_lbl, t_lbl)

# upper-right legend — color-dot + name for both samplers
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

# leapfrog HMC: resample momentum every L steps; reflect at the walls instead of rejecting
def _simulate_leapfrog(
    x0, dV, n_refresh, L_steps, eps, beta, seed,
    x_lo=X_MIN + 0.05, x_hi=X_MAX - 0.05,
):
    rng = np.random.default_rng(seed)
    x = float(x0)
    sigma_p = 1.0 / float(np.sqrt(beta))  # p ~ N(0, 1/beta) since mass = 1
    xs = [x]
    for _ in range(n_refresh):
        p = float(rng.normal(0.0, sigma_p))
        for _ in range(L_steps):
            # leapfrog: half-kick, drift, half-kick
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

# unadjusted Langevin: gradient drift + sqrt(2 eps / beta) Gaussian noise per step
def _simulate_langevin(
    x0, dV, n_steps, eps, beta, seed,
    x_lo=X_MIN + 0.05, x_hi=X_MAX - 0.05,
):
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

# interpolate the trajectory at wall-clock time t — linear between trajectory samples
def _x_at_time(xs_traj, step_dt, t):
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

# returns a closure that drives a Dot along the potential curve from a ValueTracker
def _make_ball_updater(main_ax, V_scalar, xs_traj, step_dt, t_tracker):
    def _upd(mob, dt):
        t = t_tracker.get_value()
        x = _x_at_time(xs_traj, step_dt, t)
        y = V_scalar(x)
        mob.move_to(main_ax.c2p(x, y))
    return _upd

# invisible helper dot in the trace panel — TracedPath follows it to draw x(t)
def _make_trace_helper_updater(trace_ax, xs_traj, step_dt, t_tracker):
    def _upd(mob, dt):
        t = t_tracker.get_value()
        x = _x_at_time(xs_traj, step_dt, t)
        step = t / step_dt
        mob.move_to(trace_ax.c2p(step, x))
    return _upd

# shared two-panel layout used by both scenes — different potentials, same animation logic
def _run_two_panel_scene(
    scene, V_scalar, y_min, y_max, xs_hmc, xs_lan, step_dt,
    trace_y_min=None, trace_y_max=None,
):
    # both chains animate at the same wall-clock pace; HMC's length sets the matched count
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

    # single ValueTracker drives both balls and both traces — keeps them perfectly synced
    t_tracker = ValueTracker(0.0)

    # HMC ball rides the potential curve
    p_hmc = Dot(color=HMC_COLOR, radius=0.13).set_z_index(7)
    p_hmc.move_to(main_ax.c2p(float(xs_hmc[0]), V_scalar(float(xs_hmc[0]))))
    p_hmc.add_updater(_make_ball_updater(main_ax, V_scalar, xs_hmc, step_dt, t_tracker))

    # Langevin ball rides the same curve
    p_lan = Dot(color=LANGEVIN_COLOR, radius=0.13).set_z_index(7)
    p_lan.move_to(main_ax.c2p(float(xs_lan[0]), V_scalar(float(xs_lan[0]))))
    p_lan.add_updater(_make_ball_updater(main_ax, V_scalar, xs_lan, step_dt, t_tracker))

    # invisible helpers (radius ~ 0) whose paths get traced as the x(t) timeseries
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

    # beat 1: bring up the main panel — frame, labels, then the V(x) curve
    scene.play(FadeIn(main_ax), Write(main_lbls), run_time=0.6)
    scene.play(Create(curve), run_time=1.1)
    # beat 2: bring up the trace panel and legend together
    scene.play(FadeIn(trace_ax), Write(trace_lbls), FadeIn(legend), run_time=0.5)

    scene.add(trace_line_hmc, trace_line_lan, helper_hmc, helper_lan, p_hmc, p_lan)
    scene.wait(0.2)

    # beat 3: run the simulation by advancing the tracker — linear rate keeps real-time pacing
    scene.play(
        t_tracker.animate.set_value(total),
        run_time=total, rate_func=linear,
    )

    # detach updaters so the closing frame is static
    p_hmc.clear_updaters()
    p_lan.clear_updaters()
    helper_hmc.clear_updaters()
    helper_lan.clear_updaters()
    scene.wait(1.0)

# scene 1: smooth convex bowl — HMC sweeps through, Langevin wanders in
class Scene1_SmoothBasin(Scene):

    def construct(self):
        # convex regime — eps_hmc large because the curvature is gentle
        beta = 1.0
        x_start = -3.6
        eps_hmc = 0.20
        L_steps = 80
        n_refresh = 3
        xs_hmc = _simulate_leapfrog(
            x0=x_start, dV=smooth_dV, n_refresh=n_refresh, L_steps=L_steps,
            eps=eps_hmc, beta=beta, seed=13,
        )
        # match Langevin step count to HMC's so traces end at the same time
        n_matched = len(xs_hmc) - 1
        eps_lan = 0.03  # small eps so Langevin doesn't blow past the well
        xs_lan = _simulate_langevin(
            x0=x_start, dV=smooth_dV, n_steps=n_matched,
            eps=eps_lan, beta=beta, seed=19,
        )

        _run_two_panel_scene(
            self, smooth_V_scalar, SMOOTH_Y_MIN, SMOOTH_Y_MAX,
            xs_hmc, xs_lan, step_dt=0.020,
        )

# scene 2: rugged landscape — HMC clears barriers, Langevin gets trapped
class Scene2_RuggedLandscape(Scene):

    def construct(self):
        # colder regime + smaller eps — rugged curvature needs more refresh cycles
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
