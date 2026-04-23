"""Parallel-tempering animation.

    Scene1_LadderWithSwaps -- three stacked energy-landscape panels at
        beta_hot, beta_warm, beta_cold. Each panel runs a single
        Metropolis chain against the same rugged 1D landscape used by
        glass_transition and arrhenius_wall; the chains are coupled by
        periodic replica-exchange swap attempts, alternating between
        adjacent panel pairs (0,1) and (1,2). A successful swap is
        drawn as a dashed segment within each affected panel (the
        intra-panel teleport from old x to new x, which visibly crosses
        barriers the local walker could not hop) and a double-arrow
        across panels (the inter-panel exchange). Dashes and arrows
        share the same stroke weight, the same colour, and the arrows
        share the same tip length. The cold chain, otherwise
        confined to its starting valley, visits multiple basins via
        swaps.

Render pipeline (matches the other animations in this repo):

    manim scene.py Scene1_LadderWithSwaps
    ffmpeg -y -i Scene1_LadderWithSwaps.mp4 -filter_complex \
        "[0:v]split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse" \
        -loop -1 Scene1_LadderWithSwaps.gif

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
PARTICLE = RED_E       # every chain's dot + local-move trail
SWAP = PURPLE_E        # every swap indicator -- dashes and arrows

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
# and arrhenius_wall so the animations are visually continuous.
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

BETA_HOT = 0.35    # liquid, water above every peak
BETA_WARM = 1.00   # intermediate
BETA_COLD = 2.60   # glassy, water just above deepest basin


def level_from_beta(beta):
    return float(min(E_FLOOR + THERMAL_C / max(float(beta), 1e-3), LEVEL_CAP))


# ===================================================================
# Axes and landscape rendering (matches arrhenius_wall)
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
# MH hop validity + physics weight (same definitions as
# glass_transition / arrhenius_wall so the chains look identical to
# those scenes when run solo).
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


# ===================================================================
# Stateful MH walker: one accepted hop per step() call (or the
# walker stays put if no proposal lands within max_props tries).
# Matches glass_transition's _walk_hopping by supporting global
# uniform proposals + a self-avoidance repulsion term, so the
# hot panel looks like Scene2_LiquidPhase and the cold panel looks
# like Scene3_GlassyPhase of that earlier animation.
# ===================================================================

class _Walker:
    def __init__(
        self, x0, sigma, level, seed,
        global_prob=0.0, repulsion=0.0,
        x_lo=None, x_hi=None, n_bins=80,
    ):
        self.sigma = sigma
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.global_prob = global_prob
        self.repulsion = repulsion
        self.x_lo = X_MIN + 0.05 if x_lo is None else x_lo
        self.x_hi = X_MAX - 0.05 if x_hi is None else x_hi
        self.n_bins = n_bins
        self.visits = np.zeros(n_bins, dtype=float)
        self.x = float(x0)
        self.visits[self._bin(self.x)] += 1.0

    def _bin(self, xv):
        f = (xv - self.x_lo) / (self.x_hi - self.x_lo)
        return int(np.clip(f * self.n_bins, 0, self.n_bins - 1))

    def _log_weight(self, xv, vcount):
        pw = _physics_weight(xv, self.level)
        if pw <= 0.0:
            return -np.inf
        return float(np.log(pw) - self.repulsion * vcount)

    def step(self, max_props=500):
        lw_curr = self._log_weight(self.x, self.visits[self._bin(self.x)])
        for _ in range(max_props):
            if self.global_prob > 0 and self.rng.random() < self.global_prob:
                x_new = self.rng.uniform(self.x_lo, self.x_hi)
            else:
                x_new = self.x + self.rng.normal(0.0, self.sigma)
            if x_new < self.x_lo or x_new > self.x_hi:
                continue
            vnew = self.visits[self._bin(x_new)]
            lw_new = self._log_weight(x_new, vnew)
            if not np.isfinite(lw_new):
                continue
            if lw_new < lw_curr and self.rng.random() > np.exp(lw_new - lw_curr):
                continue
            if not _hop_valid(self.x, x_new, self.level):
                continue
            self.x = x_new
            self.visits[self._bin(self.x)] += 1.0
            return self.x
        return self.x

    def set_x(self, new_x):
        """Replace x directly -- used to land a swap without running
        it through the hop-validity check."""
        self.x = float(new_x)
        self.visits[self._bin(self.x)] += 1.0


# ===================================================================
# PT driver. Every swap_every steps, alternates between pair (0,1)
# and pair (1,2). A swap is feasible only if both configurations
# lie strictly under both water levels; accepted with the standard
# Boltzmann ratio
#
#     alpha = min(1, exp((beta_i - beta_j) * (E_i - E_j))).
#
# Returns trajectories[k] = [(step, x, is_swap_landing)] and the
# list of successful swap_events.
# ===================================================================

# Per-panel proposal parameters. Hot gets global-uniform proposals
# so it decorrelates quickly; cold uses only local Gaussian proposals
# so it genuinely diffuses within its starting valley.
_GLOBAL_PROBS = (0.25, 0.12, 0.00)
_REPULSIONS = (1.00, 0.80, 0.80)
_SIGMAS = (0.55, 0.30, 0.18)


def _run_pt(betas, levels, x0s, n_steps, swap_every, seed):
    walkers = [
        _Walker(
            x0s[k], _SIGMAS[k], levels[k], seed + k + 1,
            global_prob=_GLOBAL_PROBS[k], repulsion=_REPULSIONS[k],
        )
        for k in range(3)
    ]
    master_rng = np.random.default_rng(seed + 99)

    trajectories = [[(0, walkers[k].x, False)] for k in range(3)]
    swap_events = []
    pair_counter = 0

    for step in range(1, n_steps + 1):
        for k in range(3):
            walkers[k].step()
        for k in range(3):
            trajectories[k].append((step, walkers[k].x, False))

        if step % swap_every == 0:
            pair_idx = pair_counter % 2
            pair_counter += 1
            i, j = pair_idx, pair_idx + 1

            x_i, x_j = walkers[i].x, walkers[j].x
            E_i, E_j = _energy_scalar(x_i), _energy_scalar(x_j)
            # Feasibility: both configs must live under both levels.
            if E_i >= levels[j] or E_j >= levels[i]:
                continue

            delta = (betas[i] - betas[j]) * (E_i - E_j)
            if master_rng.random() < np.exp(min(0.0, delta)):
                walkers[i].set_x(x_j)
                walkers[j].set_x(x_i)
                trajectories[i][-1] = (step, walkers[i].x, True)
                trajectories[j][-1] = (step, walkers[j].x, True)
                swap_events.append((step, i, j, walkers[i].x, walkers[j].x))

    return trajectories, swap_events


# ===================================================================
# Per-chain trail + moving dot.
#
# MH step  -> solid RED_E segment that grows from (x_k, E(x_k)) out
#             to the particle's live position.
# swap land -> dashed RED_C segment that appears at the swap instant;
#             it connects (x_old, E(x_old)) to (x_new, E(x_new)) in
#             the same panel and is free to cross the curve.
#
# The dashed segments share their stroke weight with the cross-panel
# swap arrows below.
# ===================================================================

SWAP_STROKE_WIDTH = 3.0
SWAP_DASH_LENGTH = 0.14
SWAP_ARROW_TIP_LENGTH = 0.22


def _build_trail_and_particle(ax, traj, step_dt,
                               dot_radius=0.10, stroke_width=2.6):
    xs_arr = [t[1] for t in traj]
    is_swap = [t[2] for t in traj]
    n = len(xs_arr)
    pts = [ax.c2p(x, _energy_scalar(x)) for x in xs_arr]

    segments = []
    for i in range(n - 1):
        if is_swap[i + 1]:
            seg = DashedLine(
                pts[i], pts[i + 1],
                color=SWAP,
                stroke_width=SWAP_STROKE_WIDTH,
                dash_length=SWAP_DASH_LENGTH,
            ).set_z_index(3)
        else:
            seg = Line(
                pts[i], pts[i + 1],
                color=PARTICLE,
                stroke_width=stroke_width,
            ).set_z_index(3)
        seg.set_stroke(opacity=0.0)
        segments.append(seg)

    container = VGroup(*segments)
    particle = Dot(
        color=PARTICLE, radius=dot_radius,
    ).move_to(pts[0]).set_z_index(16)

    t_state = {"t": 0.0, "revealed": 0}
    p_state = {"t": 0.0}
    total = (n - 1) * step_dt

    def trail_updater(mob, dt):
        t_state["t"] += dt
        tt = min(t_state["t"], total)
        idx_f = tt / step_dt
        idx = int(idx_f)
        frac = idx_f - idx

        while t_state["revealed"] < min(idx, n - 1):
            r = t_state["revealed"]
            seg = segments[r]
            if is_swap[r + 1]:
                seg.set_stroke(opacity=1.0)
            else:
                start = np.array(pts[r])
                end = np.array(pts[r + 1])
                if not np.allclose(start, end):
                    seg.set_stroke(opacity=1.0)
                    seg.put_start_and_end_on(start, end)
            t_state["revealed"] = r + 1

        if idx < n - 1:
            seg = segments[idx]
            if is_swap[idx + 1]:
                if frac > 0.30:
                    seg.set_stroke(opacity=1.0)
            else:
                start = np.array(pts[idx])
                end = np.array(pts[idx + 1])
                if not np.allclose(start, end):
                    seg.set_stroke(opacity=1.0)
                    f = max(1e-3, min(1.0, frac))
                    seg.put_start_and_end_on(
                        start, start + (end - start) * f,
                    )

    def particle_updater(mob, dt):
        p_state["t"] += dt
        tt = min(p_state["t"], total)
        idx_f = tt / step_dt
        idx = int(idx_f)
        frac = idx_f - idx
        if idx >= n - 1:
            mob.move_to(pts[-1])
        else:
            if is_swap[idx + 1]:
                if frac < 0.5:
                    mob.move_to(pts[idx])
                else:
                    mob.move_to(pts[idx + 1])
            else:
                start = np.array(pts[idx])
                end = np.array(pts[idx + 1])
                f = max(0.0, min(1.0, frac))
                mob.move_to(start + (end - start) * f)

    return container, particle, trail_updater, particle_updater, total


# ===================================================================
# Cross-panel swap arrow: a RED_C double-arrow from one particle's
# post-swap position to the other's, pulsing briefly around the swap
# instant. All arrows share the same stroke_width and tip_length so
# they read as a single recurring symbol.
# ===================================================================

def _make_cross_panel_arrow(ax_i, ax_j, x_i_new, x_j_new):
    start = ax_i.c2p(x_i_new, _energy_scalar(x_i_new))
    end = ax_j.c2p(x_j_new, _energy_scalar(x_j_new))
    arrow = DoubleArrow(
        start, end,
        color=SWAP,
        stroke_width=SWAP_STROKE_WIDTH,
        buff=0.16,
        tip_length=SWAP_ARROW_TIP_LENGTH,
    ).set_z_index(14)
    arrow.set_stroke(opacity=0.0)
    arrow.set_fill(opacity=0.0)
    return arrow


def _swap_arrow_updater(arrow_schedule, step_dt,
                        fade_in=0.22, hold=0.55, fade_out=0.35):
    """Updater that pulses each arrow around its swap instant. The
    arrow appears at (step * step_dt) - fade_in, stays full for
    `hold` seconds, then fades out over `fade_out` seconds."""
    state = {"t": 0.0}
    total_life = fade_in + hold + fade_out

    def updater(mob, dt):
        state["t"] += dt
        tt = state["t"]
        for (step, arrow) in arrow_schedule:
            t_start = step * step_dt - fade_in
            dt_local = tt - t_start
            if dt_local < 0 or dt_local > total_life:
                arrow.set_stroke(opacity=0.0)
                arrow.set_fill(opacity=0.0)
                continue
            if dt_local < fade_in:
                alpha = dt_local / fade_in
            elif dt_local < fade_in + hold:
                alpha = 1.0
            else:
                alpha = 1.0 - (dt_local - fade_in - hold) / fade_out
            alpha = max(0.0, min(1.0, alpha))
            arrow.set_stroke(opacity=alpha)
            arrow.set_fill(opacity=alpha)

    return updater


# ===================================================================
# Simulation budget. Tuned so the cold chain's swap ladder carries
# it between basins several times in the clip.
# ===================================================================

N_STEPS = 120
STEP_DT = 0.10
SWAP_EVERY = 6
SEED = 404


# ===================================================================
# Scene 1 -- three stacked PT panels, no title, no caption
# ===================================================================

class Scene1_LadderWithSwaps(Scene):
    def construct(self):
        betas = [BETA_HOT, BETA_WARM, BETA_COLD]
        levels = [level_from_beta(b) for b in betas]
        # Match the labels used in arrhenius_wall so the two scenes sit
        # side-by-side in the deck without a label mismatch: the three
        # rows correspond to high / critical / low temperature.
        beta_tex = [
            r"\beta_{\mathrm{high}}",
            r"\beta_c",
            r"\beta_{\mathrm{low}}",
        ]
        temp_tex = [
            r"$T \gg T_c$",
            r"$T \approx T_c$",
            r"$T \ll T_c$",
        ]

        axes_list, curves, fills, surfaces = [], [], [], []
        beta_lbls, temp_lbls = [], []
        for k in range(3):
            ax = _panel_axes(PANEL_Y[k])
            curve = _panel_curve(ax)
            wf = _water_fill(ax, levels[k])
            ws = _water_surface(ax, levels[k])
            bl = MathTex(beta_tex[k], color=FG, font_size=EQN_SIZE)
            bl.next_to(ax, LEFT, buff=0.30)
            tl = Tex(temp_tex[k], color=FG, font_size=LABEL_SIZE)
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

        # Cold chain starts at the deepest submerged point; other
        # chains start there too so the only thing that separates
        # their trails is their sampler dynamics + swaps.
        ys_masked = np.where(_PROBE_YS < levels[2], _PROBE_YS, np.inf)
        cold_center = float(_PROBE_XS[int(np.argmin(ys_masked))])
        x0s = [cold_center, cold_center, cold_center]

        trajectories, swap_events = _run_pt(
            betas=betas, levels=levels, x0s=x0s,
            n_steps=N_STEPS, swap_every=SWAP_EVERY, seed=SEED,
        )

        trails, particles = [], []
        t_upds, p_upds = [], []
        total_time = 0.0
        for k in range(3):
            tr, p, t_up, p_up, total = _build_trail_and_particle(
                axes_list[k], trajectories[k], STEP_DT,
                dot_radius=0.10, stroke_width=2.6,
            )
            trails.append(tr)
            particles.append(p)
            t_upds.append(t_up)
            p_upds.append(p_up)
            total_time = max(total_time, total)

        # Swap arrows: each successful swap gets one RED_C double-
        # arrow connecting the two particles' new positions across
        # the two panels. They are pre-built and pulsed by a single
        # updater.
        arrow_schedule = []
        for (step, i, j, x_i_new, x_j_new) in swap_events:
            arr = _make_cross_panel_arrow(
                axes_list[i], axes_list[j], x_i_new, x_j_new,
            )
            arrow_schedule.append((step, arr))
        arrow_group = VGroup(*[a for (_, a) in arrow_schedule])
        arrow_upd = _swap_arrow_updater(arrow_schedule, STEP_DT)

        for tr, p in zip(trails, particles):
            self.add(tr, p)
        self.add(arrow_group)
        self.wait(0.3)

        for tr, t_up in zip(trails, t_upds):
            tr.add_updater(t_up)
        for p, p_up in zip(particles, p_upds):
            p.add_updater(p_up)
        arrow_group.add_updater(arrow_upd)

        self.wait(total_time + 0.5)

        for tr, t_up in zip(trails, t_upds):
            tr.remove_updater(t_up)
        for p, p_up in zip(particles, p_upds):
            p.remove_updater(p_up)
        arrow_group.remove_updater(arrow_upd)

        self.wait(1.4)
