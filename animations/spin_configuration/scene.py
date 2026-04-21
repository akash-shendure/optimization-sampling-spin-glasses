"""Ising / Edwards-Anderson spin-configuration animations.

Six scenes that share a board layout and chain end-to-end as consecutive
slides:

    Scene1_Lattice              — grow the lattice as a tree from the top-left
                                  node outward; write H(s) = -J Σ s_i s_j.
    Scene2_EdgeColoring         — color edges (blue = same-spin, red = opposite)
                                  and swap the summation for -J (N_blue - N_red).
    Scene3_Relaxation           — single-spin Metropolis drives the ferromagnet
                                  to the all-up ground state; integer counts in
                                  the Hamiltonian update with each flip.
    Scene4_FlipSymmetry         — flip one BFS diagonal at a time to reach the
                                  Z/2-reflected ground state.
    Scene5_EA_InitCouplings     — Edwards-Anderson: swap the equation to the
                                  disordered form and recolor every edge by
                                  J_{ij} ~ N(J_0, J^2) using an HSL diverging
                                  gradient (J → +∞ = BLUE_E, J → -∞ = RED_E).
    Scene6_EA_Evolution         — Metropolis on the EA couplings; edge colors
                                  are held constant (couplings don't change).

All blue↔red transitions (including Scenes 3/4) use the same HSL diverging
interpolation that passes through white instead of muddy RGB midpoints.

Render with:

    manim scene.py Scene1_Lattice
    manim scene.py Scene2_EdgeColoring
    manim scene.py Scene3_Relaxation
    manim scene.py Scene4_FlipSymmetry
    manim scene.py Scene5_EA_InitCouplings
    manim scene.py Scene6_EA_Evolution

Videos land next to this file at 1080p / 50 fps.
"""
from pathlib import Path
import colorsys

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

# ---------- lattice ----------
ROWS = 5
COLS = 6
SPACING = 1.05
ARROW_LEN = 0.56
CIRCLE_R = 0.34

# ---------- palette ----------
FG = BLACK
GRID = "#555555"
_BLUE_HEX = "#1C758A"  # manim BLUE_E
_RED_HEX = "#CF5044"   # manim RED_E

EQN_SIZE = 44
BOARD_SHIFT = UP * 0.35

# ---------- pacing ----------
FLIP_RUNTIME = 0.24
DIAGONAL_RUNTIME = 0.32
TREE_LAYER_RUNTIME = 0.32

# ---------- ferromagnetic Scene 3 greedy descent ----------
INIT_SEED = 1
INIT_BIAS = 0.60

# ---------- Edwards-Anderson (Scenes 5–6) ----------
EA_J0 = 0.0
EA_JV = 1.0          # J in N(J_0, J^2)
EA_COUPLING_SEED = 17
COUPLING_SCALE = 1.3  # tanh scale: smaller = more saturated for typical |J|

# "Saturation" J value used to paint a binary blue/red endpoint with the
# same gradient function the EA scenes use.
SAT_J = 8.0

# ---------- SK model (Scenes 7–8) ----------
SK_N = 7
SK_RADIUS = 2.2

# ---------- Sparse random graph (Scenes 9–10) ----------
SRG_SEED = 123   # which edges survive the K_7 → SRG pruning
SRG_K = 10       # edges kept (avg degree 20/7 ≈ 2.86 ≈ c = 3)


# ===================================================================
# Geometry / lattice helpers
# ===================================================================

def _positions():
    pos = {}
    for i in range(ROWS):
        for j in range(COLS):
            x = (j - (COLS - 1) / 2) * SPACING
            y = ((ROWS - 1) / 2 - i) * SPACING
            pos[(i, j)] = np.array([x, y, 0.0])
    return pos


def _edge_keys():
    keys = []
    for i in range(ROWS):
        for j in range(COLS):
            if j + 1 < COLS:
                keys.append(((i, j), (i, j + 1)))
            if i + 1 < ROWS:
                keys.append(((i, j), (i + 1, j)))
    return keys


def _layers():
    groups = {}
    for i in range(ROWS):
        for j in range(COLS):
            groups.setdefault(i + j, []).append((i, j))
    return [groups[k] for k in sorted(groups)]


def _spin_arrow(center, spin):
    d = UP if spin > 0 else DOWN
    start = center - d * ARROW_LEN / 2
    end = center + d * ARROW_LEN / 2
    return Arrow(
        start, end, buff=0, color=FG, stroke_width=5,
        max_tip_length_to_length_ratio=0.38,
        max_stroke_width_to_length_ratio=14,
    )


def _node_circle(center):
    return Circle(
        radius=CIRCLE_R, color=FG,
        fill_color=WHITE, fill_opacity=1.0,
        stroke_width=1.8,
    ).move_to(center)


# ===================================================================
# Diverging color gradient (HSL through white midpoint)
# ===================================================================

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )


def _coupling_color(j_val, scale=COUPLING_SCALE):
    """Map a coupling strength J → hex color via tanh + HSL diverging.

    J → +∞ saturates to BLUE_E, J → -∞ to RED_E, J = 0 renders white.
    Hue is held constant on each side; lightness and saturation interpolate
    between white (t=0) and the saturated endpoint (|t|=1). This gives a
    clean diverging palette without the muddy midtones of linear-RGB mixing.
    """
    t = float(np.tanh(j_val / scale))
    base_hex = _BLUE_HEX if t >= 0 else _RED_HEX
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    abs_t = abs(t)
    new_l = 1.0 - abs_t * (1.0 - l)
    new_s = abs_t * s
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s))


def _diverging_edge_anim(line, from_j, to_j):
    """Animate a line's stroke color along the HSL diverging gradient by
    linearly interpolating J from from_j to to_j. Passing through J = 0
    routes the color through white, which is the point when coupling
    strength can legitimately land near zero (Scenes 5-6)."""
    def update(m, alpha):
        j = from_j + alpha * (to_j - from_j)
        m.set_stroke(color=_coupling_color(j))
    return UpdateFromAlphaFunc(line, update)


def _edge_color(s1, s2):
    """Binary edge color based on spin product — same gradient endpoints
    used for the EA couplings, just saturated at ±SAT_J."""
    return _coupling_color(s1 * s2 * SAT_J)


def _binary_J(sign):
    return float(sign) * SAT_J


# ===================================================================
# Ferromagnetic state / dynamics (Scenes 1–4)
# ===================================================================

def _initial_spins():
    rng = np.random.default_rng(INIT_SEED)
    return rng.choice([-1, 1], size=(ROWS, COLS), p=[1 - INIT_BIAS, INIT_BIAS])


def _count_colors(spins):
    blue = 0
    red = 0
    for (a, b) in _edge_keys():
        if spins[a] == spins[b]:
            blue += 1
        else:
            red += 1
    return blue, red


def _ferro_J():
    """Uniform J = +1 for the 2-D ferromagnet."""
    return {k: 1.0 for k in _edge_keys()}


def _greedy_descent_lattice(initial, J, max_flips=400):
    """Deterministic steepest descent on the 2-D grid: each step flips the
    single spin with the most-negative ΔE. Ties broken by lowest (i, j).
    Stops at a strict local minimum. Because energy strictly decreases,
    a spin that just flipped can never be the argmax-ΔE next step."""
    spins = initial.copy().astype(int)
    flips = []
    while len(flips) < max_flips:
        best = None  # (dE, i, j)
        for i in range(ROWS):
            for j in range(COLS):
                h = 0.0
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ROWS and 0 <= nj < COLS:
                        key = (min((i, j), (ni, nj)), max((i, j), (ni, nj)))
                        h += J[key] * int(spins[ni, nj])
                dE = 2.0 * int(spins[i, j]) * h
                if dE < 0 and (best is None or dE < best[0]):
                    best = (dE, i, j)
        if best is None:
            break
        _, i, j = best
        spins[i, j] *= -1
        flips.append((i, j))
    return spins, flips


# ===================================================================
# Edwards-Anderson couplings (Scenes 5–7)
# ===================================================================

def _ea_couplings():
    """Sample J_{ij} ~ N(J_0, J^2) for every lattice edge (deterministic)."""
    rng = np.random.default_rng(EA_COUPLING_SEED)
    return {k: float(rng.normal(EA_J0, EA_JV)) for k in _edge_keys()}


# ===================================================================
# Board construction
# ===================================================================

def _build_board(spins, colored_edges=False):
    pos = _positions()
    edges = {}
    for k in _edge_keys():
        a, b = k
        pa, pb = pos[a], pos[b]
        unit = (pb - pa) / np.linalg.norm(pb - pa)
        start = pa + unit * CIRCLE_R
        end = pb - unit * CIRCLE_R
        color = _edge_color(spins[a], spins[b]) if colored_edges else GRID
        width = 4.5 if colored_edges else 2.4
        line = Line(start, end, color=color, stroke_width=width)
        line.set_z_index(0)
        edges[k] = line
    circles = {}
    arrows = {}
    for k, p in pos.items():
        c = _node_circle(p)
        c.set_z_index(1)
        circles[k] = c
        a_m = _spin_arrow(p, int(spins[k]))
        a_m.set_z_index(2)
        arrows[k] = a_m
    edges_g = VGroup(*edges.values())
    circles_g = VGroup(*circles.values())
    arrows_g = VGroup(*arrows.values())
    board = VGroup(edges_g, circles_g, arrows_g).move_to(ORIGIN).shift(BOARD_SHIFT)
    shifted_pos = {k: p + BOARD_SHIFT for k, p in pos.items()}
    return shifted_pos, edges, circles, arrows, board


# ===================================================================
# Equations
# ===================================================================

def _eqn_signed():
    return MathTex(
        r"H(s)\,=\,-J",
        r"\sum_{(i,j)\in E} s_i\,s_j",
        r",\;\, J>0",
        color=FG, font_size=EQN_SIZE,
    ).to_edge(DOWN, buff=0.55)


def _eqn_count(blue, red):
    e = MathTex(
        r"H(s)\,=\,-J",
        r"\bigl(",
        str(blue),
        r"-",
        str(red),
        r"\bigr)",
        color=FG, font_size=EQN_SIZE,
    )
    e[2].set_color(_BLUE_HEX)
    e[4].set_color(_RED_HEX)
    return e.to_edge(DOWN, buff=0.55)


def _eqn_ea():
    return MathTex(
        r"H(s)\,=\,-\sum_{(i,j)\in E} J_{ij}\, s_i\, s_j,\;\;",
        r"J_{ij} \sim \mathcal{N}(J_0, J^2)",
        color=FG, font_size=EQN_SIZE - 4,
    ).to_edge(DOWN, buff=0.55)


# ===================================================================
# Scenes
# ===================================================================

class Scene1_Lattice(Scene):
    """Tree-grow the lattice from the top-left node outward at uniform speed."""

    def construct(self):
        spins = _initial_spins()
        pos, edges, circles, arrows, board = _build_board(spins)
        eqn = _eqn_signed()

        layers = _layers()
        layer_groups = []
        start_cell = layers[0][0]  # (0, 0) — top-left
        layer_groups.append(
            AnimationGroup(
                FadeIn(circles[start_cell], scale=0.55),
                GrowArrow(arrows[start_cell]),
            )
        )
        for L in range(1, len(layers)):
            cells = layers[L]
            incoming = set()
            for (i, j) in cells:
                for parent in [(i - 1, j), (i, j - 1)]:
                    if 0 <= parent[0] < ROWS and 0 <= parent[1] < COLS:
                        incoming.add((parent, (i, j)))
            anims = [Create(edges[k]) for k in incoming]
            anims += [FadeIn(circles[c], scale=0.55) for c in cells]
            anims += [GrowArrow(arrows[c]) for c in cells]
            layer_groups.append(AnimationGroup(*anims))

        # lag_ratio < 1 keeps the wavefront flowing between layers.
        self.play(LaggedStart(*layer_groups, lag_ratio=0.45), run_time=3.0)
        self.play(Write(eqn), run_time=0.9)
        self.wait(1.0)


class Scene2_EdgeColoring(Scene):
    """Color every edge; transform the summation into -J (N_blue - N_red)."""

    def construct(self):
        spins = _initial_spins()
        pos, edges, circles, arrows, board = _build_board(spins)
        eqn_old = _eqn_signed()

        self.add(board, eqn_old)
        self.wait(0.3)

        recolor = [
            line.animate.set_stroke(color=_edge_color(spins[a], spins[b]), width=4.5)
            for (a, b), line in edges.items()
        ]
        self.play(LaggedStart(*recolor, lag_ratio=0.035), run_time=1.6)

        blue, red = _count_colors(spins)
        eqn_new = _eqn_count(blue, red)
        self.play(TransformMatchingTex(eqn_old, eqn_new), run_time=1.0)
        self.wait(1.0)


class Scene3_Relaxation(Scene):
    """Ferromagnetic Metropolis relaxation to the all-up ground state."""

    def construct(self):
        spins = _initial_spins().copy()
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.5)

        _, flips = _greedy_descent_lattice(spins, _ferro_J())
        current = spins.copy()
        for (i, j) in flips:
            current[i, j] *= -1
            edge_anims = []
            for (a, b) in _edge_keys():
                if (i, j) == a or (i, j) == b:
                    new_sign = int(current[a]) * int(current[b])
                    # Binary flip: route blue ↔ red through the HSL diverging
                    # gradient's zero (white) midpoint.
                    edge_anims.append(
                        _diverging_edge_anim(
                            edges[(a, b)], _binary_J(-new_sign), _binary_J(new_sign)
                        )
                    )
            blue, red = _count_colors(current)
            new_eqn = _eqn_count(blue, red)
            self.play(
                Rotate(arrows[(i, j)], PI, about_point=pos[(i, j)]),
                *edge_anims,
                TransformMatchingTex(current_eqn, new_eqn),
                run_time=FLIP_RUNTIME,
            )
            current_eqn = new_eqn
        self.wait(1.2)


class Scene4_FlipSymmetry(Scene):
    """Flip one BFS diagonal at a time to reach the Z/2-reflected ground state."""

    def construct(self):
        spins = np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.4)

        current = spins.copy()
        for layer_cells in _layers():
            rotates = []
            for (i, j) in layer_cells:
                current[i, j] = -1
                rotates.append(Rotate(arrows[(i, j)], PI, about_point=pos[(i, j)]))
            affected = set()
            for (i, j) in layer_cells:
                for (a, b) in _edge_keys():
                    if (i, j) == a or (i, j) == b:
                        affected.add((a, b))
            edge_anims = []
            for key in affected:
                new_sign = int(current[key[0]]) * int(current[key[1]])
                edge_anims.append(
                    _diverging_edge_anim(
                        edges[key], _binary_J(-new_sign), _binary_J(new_sign)
                    )
                )
            blue, red = _count_colors(current)
            new_eqn = _eqn_count(blue, red)
            self.play(
                *rotates,
                *edge_anims,
                TransformMatchingTex(current_eqn, new_eqn),
                run_time=DIAGONAL_RUNTIME,
            )
            current_eqn = new_eqn
        self.wait(1.0)


class Scene5_EA_InitCouplings(Scene):
    """Initialize the Edwards-Anderson disorder: swap the equation and recolor
    every edge from saturated blue to its sampled J_ij via the gradient."""

    def construct(self):
        # Starting state matches Scene 4's end: all spins down, saturated-blue edges.
        spins = -np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.4)

        ea_eqn = _eqn_ea()
        self.play(
            FadeOut(current_eqn, shift=UP * 0.12),
            FadeIn(ea_eqn, shift=UP * 0.12),
            run_time=0.9,
        )
        self.wait(0.2)

        J = _ea_couplings()
        # Group edges by max-endpoint BFS layer so the wave advances by
        # diagonal, matching Scene 1's tree growth and Scene 4's flip wave.
        by_layer = {}
        for k in _edge_keys():
            a, b = k
            by_layer.setdefault(max(a[0] + a[1], b[0] + b[1]), []).append(k)
        diagonal_groups = []
        for L in sorted(by_layer):
            anims = [
                _diverging_edge_anim(edges[k], from_j=_binary_J(1), to_j=J[k])
                for k in by_layer[L]
            ]
            diagonal_groups.append(AnimationGroup(*anims))
        self.play(
            LaggedStart(*diagonal_groups, lag_ratio=0.45),
            run_time=3.0,
        )
        self.wait(1.0)


class Scene6_EA_Evolution(Scene):
    """Metropolis dynamics on the EA lattice. Edge colors stay fixed — only
    the spins evolve under the frozen couplings."""

    def construct(self):
        spins = -np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        # Overwrite the uniform-blue edges with EA coupling colors (matches
        # the final state of Scene 5).
        J = _ea_couplings()
        for k in _edge_keys():
            edges[k].set_stroke(color=_coupling_color(J[k]))
        eqn = _eqn_ea()

        self.add(board, eqn)
        self.wait(0.4)

        _, flips = _greedy_descent_lattice(spins, J)
        for (i, j) in flips:
            self.play(
                Rotate(arrows[(i, j)], PI, about_point=pos[(i, j)]),
                run_time=FLIP_RUNTIME,
            )
        self.wait(1.2)


def _sk_positions(center=None):
    """7 ring vertices on a circle of radius SK_RADIUS, with a vertex at the top."""
    if center is None:
        center = BOARD_SHIFT
    positions = {}
    for v in range(SK_N):
        angle = 2 * np.pi * v / SK_N + np.pi / 2  # vertex 0 at TOP
        positions[v] = (
            np.array([SK_RADIUS * np.cos(angle), SK_RADIUS * np.sin(angle), 0.0])
            + center
        )
    return positions


def _sk_line_endpoints(pa, pb):
    """Shorten a line between two SK nodes so it terminates at each circle
    boundary; collapse to a point for self-loops."""
    d = pb - pa
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return pa.copy(), pa.copy()
    unit = d / L
    return pa + unit * CIRCLE_R, pb - unit * CIRCLE_R


def _world_lattice_positions():
    """Lattice positions already shifted by BOARD_SHIFT (world coordinates)."""
    return {k: p + BOARD_SHIFT for k, p in _positions().items()}


def _sk_edge_assignment():
    """Hungarian-match each of the 49 lattice edges to a unique ordered SK
    pair (49 = 7×7 including self-pairs), minimizing total endpoint
    displacement. Returns {lattice_edge_key: (v_a, v_b)} where v_a, v_b are
    chosen/ordered so the lattice edge's endpoints map to v_a and v_b with
    minimum sum of distances."""
    from scipy.optimize import linear_sum_assignment

    keys = _edge_keys()
    lat_pos = _world_lattice_positions()
    sk_pos = _sk_positions()
    ordered_pairs = [(va, vb) for va in range(SK_N) for vb in range(SK_N)]
    n = len(keys)
    cost = np.zeros((n, n))
    swap = np.zeros((n, n), dtype=bool)
    for i, k in enumerate(keys):
        pa, pb = lat_pos[k[0]], lat_pos[k[1]]
        for j, (va, vb) in enumerate(ordered_pairs):
            qa, qb = sk_pos[va], sk_pos[vb]
            c_direct = np.linalg.norm(pa - qa) + np.linalg.norm(pb - qb)
            c_swap = np.linalg.norm(pa - qb) + np.linalg.norm(pb - qa)
            if c_direct <= c_swap:
                cost[i, j] = c_direct
            else:
                cost[i, j] = c_swap
                swap[i, j] = True
    row, col = linear_sum_assignment(cost)
    out = {}
    for i in range(n):
        k = keys[row[i]]
        va, vb = ordered_pairs[col[i]]
        if swap[row[i], col[i]]:
            va, vb = vb, va
        out[k] = (va, vb)
    return out


def _sk_node_assignment():
    """Each of the 7 SK vertices picks the nearest distinct lattice vertex to
    be moved into its place. Returns {sk_vertex: lattice_key}."""
    from scipy.optimize import linear_sum_assignment

    lat_pos = _world_lattice_positions()
    sk_pos = _sk_positions()
    lat_keys = list(lat_pos.keys())
    cost = np.zeros((SK_N, len(lat_keys)))
    for v in range(SK_N):
        for idx, lk in enumerate(lat_keys):
            cost[v, idx] = np.linalg.norm(sk_pos[v] - lat_pos[lk])
    row, col = linear_sum_assignment(cost)
    return {int(row[i]): lat_keys[col[i]] for i in range(SK_N)}


def _sk_couplings_from_lattice(edge_to_pair, lat_J):
    """For each canonical SK pair (a,b) with a<b, adopt the coupling J from
    the *surviving* lattice edge — i.e. the later-added one (higher k_idx),
    which is rendered in front and therefore kept by Scene 7's removal pass.
    Inserts in forward _edge_keys() order so dict iteration matches the
    on-screen edge stacking order at the end of Scene 7."""
    # First pass: identify the surviving (highest-k_idx) lattice edge per pair.
    survivor_k = {}
    for k in _edge_keys():
        va, vb = edge_to_pair[k]
        if va == vb:
            continue
        canonical = (min(va, vb), max(va, vb))
        survivor_k[canonical] = k
    # Second pass: insert in forward k order, keeping only the survivor.
    sk_J = {}
    for k in _edge_keys():
        va, vb = edge_to_pair[k]
        if va == vb:
            continue
        canonical = (min(va, vb), max(va, vb))
        if survivor_k[canonical] == k:
            sk_J[canonical] = lat_J[k]
    return sk_J


def _local_field_ring(spins, J, i, n):
    h = 0.0
    for j in range(n):
        if j == i:
            continue
        pair = (min(i, j), max(i, j))
        if pair in J:
            h += J[pair] * int(spins[j])
    return h


def _greedy_descent_ring(initial, J, max_flips=200):
    """Deterministic steepest descent on a spin system defined by canonical-
    pair couplings J. Each step flips the spin with the most-negative ΔE
    (ties broken by lowest index). Stops at a strict local minimum."""
    spins = np.array(initial, dtype=int)
    n = len(spins)
    flips = []
    while len(flips) < max_flips:
        best = None  # (dE, i)
        for i in range(n):
            dE = 2.0 * int(spins[i]) * _local_field_ring(spins, J, i, n)
            if dE < 0 and (best is None or dE < best[0]):
                best = (dE, i)
        if best is None:
            break
        _, i = best
        spins[i] *= -1
        flips.append(i)
    return spins, flips


def _srg_edges(seed=SRG_SEED, k=SRG_K):
    """Select k random canonical SK pairs to survive the K_7 → sparse random
    graph pruning. Remaining 21-k edges are dropped."""
    all_pairs = [(a, b) for a in range(SK_N) for b in range(a + 1, SK_N)]
    rng = np.random.default_rng(seed)
    rng.shuffle(all_pairs)
    return set(all_pairs[:k])


def _move_rotate_arrow(arrow, from_pos, to_pos, initial_spin, final_spin):
    """Translate an arrow's center linearly from from_pos to to_pos while
    rotating it smoothly from initial_spin orientation to final_spin.
    Reconstructs endpoints each frame so the arrow stays centered on the
    linear trajectory (unlike Transform with path_arc which curves the
    center path outside the moving circle)."""
    angle_start = 0.0 if initial_spin > 0 else PI
    angle_end = 0.0 if final_spin > 0 else PI

    def update(m, alpha):
        center = from_pos + alpha * (to_pos - from_pos)
        angle = angle_start + alpha * (angle_end - angle_start)
        d = np.array([np.sin(angle), np.cos(angle), 0.0])
        start = center - d * ARROW_LEN / 2
        end = center + d * ARROW_LEN / 2
        m.put_start_and_end_on(start, end)

    return UpdateFromAlphaFunc(arrow, update)


class Scene7_SK_Transition(Scene):
    """Transition from the EA lattice to K_7 on a ring. Each lattice edge is
    Hungarian-assigned to a unique ordered SK pair (minimising total endpoint
    travel), and 7 lattice vertices Hungarian-assigned (by nearest distance)
    to the SK ring slide into place and rotate to down. Remaining 23 vertices
    fade out. Self-loops and the behind-duplicates of each canonical SK pair
    fade out, leaving K_7 with 21 edges."""

    def construct(self):
        J = _ea_couplings()
        start_spins = -np.ones((ROWS, COLS), dtype=int)
        final_spins, _ = _greedy_descent_lattice(start_spins, J)

        pos, edges, circles, arrows, board = _build_board(final_spins, colored_edges=True)
        for k in _edge_keys():
            edges[k].set_stroke(color=_coupling_color(J[k]))
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.4)

        sk_pos = _sk_positions()
        edge_to_pair = _sk_edge_assignment()
        sk_to_lat = _sk_node_assignment()
        surviving_lat_keys = set(sk_to_lat.values())
        dying_lat_keys = [k for k in pos if k not in surviving_lat_keys]

        # --- Edge morphs: each lattice edge → its assigned SK ordered pair ---
        edge_morphs = []
        for k in _edge_keys():
            va, vb = edge_to_pair[k]
            color = _coupling_color(J[k])
            start, end = _sk_line_endpoints(sk_pos[va], sk_pos[vb])
            target = Line(start, end, color=color, stroke_width=4.5)
            target.set_z_index(edges[k].z_index)
            edge_morphs.append(Transform(edges[k], target))

        # --- Node animations ---
        # Circles translate linearly along the straight segment from lattice
        # to ring. Arrows do the same translation AND rotate into the "down"
        # orientation (if not already there), both driven by the *same*
        # linear trajectory so the arrow stays centered in its circle.
        lat_world_pos = _world_lattice_positions()
        node_anims = []
        for v in range(SK_N):
            lat_k = sk_to_lat[v]
            from_pos = lat_world_pos[lat_k]
            target_pos = sk_pos[v]
            node_anims.append(circles[lat_k].animate.move_to(target_pos))
            current_spin = int(final_spins[lat_k])
            node_anims.append(
                _move_rotate_arrow(
                    arrows[lat_k], from_pos, target_pos,
                    initial_spin=current_spin, final_spin=-1,
                )
            )

        dying_group = VGroup(
            *[circles[k] for k in dying_lat_keys],
            *[arrows[k] for k in dying_lat_keys],
        )

        self.play(
            *edge_morphs,
            *node_anims,
            FadeOut(dying_group),
            run_time=2.4,
        )

        # --- Fade out self-loops and the behind-duplicate of each pair ---
        # Reverse iteration keeps the LAST-added (front-most) lattice edge for
        # each canonical SK pair and drops the earlier (behind) one.
        to_remove = []
        seen = set()
        for k in reversed(_edge_keys()):
            va, vb = edge_to_pair[k]
            if va == vb:
                to_remove.append(edges[k])
                continue
            canonical = (min(va, vb), max(va, vb))
            if canonical in seen:
                to_remove.append(edges[k])
            else:
                seen.add(canonical)

        self.play(*[FadeOut(e) for e in to_remove], run_time=0.9)
        self.wait(1.0)


def _build_ring_board(spins, J_all, kept_pairs=None):
    """Build ring lattice: circles + arrows at SK_N ring positions, plus
    edges (shortened to circle boundaries) for each canonical pair in
    `kept_pairs` (defaults to every pair in J_all) colored by J_all value.
    Edges are added in J_all key order (filtered by kept_pairs membership),
    so the on-screen stacking order matches the previous scene's surviving
    edges and visual continuity is preserved across the 7→8 and 9→10 cuts."""
    sk_pos = _sk_positions()
    circles = {}
    arrows = {}
    for v in range(SK_N):
        c = _node_circle(sk_pos[v]); c.set_z_index(1)
        circles[v] = c
        a_m = _spin_arrow(sk_pos[v], int(spins[v])); a_m.set_z_index(2)
        arrows[v] = a_m
    if kept_pairs is None:
        ordered = list(J_all.keys())
    else:
        kept_set = set(kept_pairs)
        ordered = [c for c in J_all.keys() if c in kept_set]
    edges = {}
    for canonical in ordered:
        va, vb = canonical
        start, end = _sk_line_endpoints(sk_pos[va], sk_pos[vb])
        line = Line(start, end, color=_coupling_color(J_all[canonical]), stroke_width=4.5)
        line.set_z_index(0)
        edges[canonical] = line
    board = VGroup(*edges.values(), *circles.values(), *arrows.values())
    return sk_pos, edges, circles, arrows, board


def _sk_scene_end_spins(sk_J):
    """Compute the spin configuration at the end of Scene 8 (zero-T descent
    from all-down on the SK couplings) so Scene 9 can match continuously."""
    final, _ = _greedy_descent_ring([-1] * SK_N, sk_J)
    return final.tolist()


class Scene8_SK_Evolution(Scene):
    """Strict zero-T descent on the SK heptagon until no single-spin flip
    lowers the energy. No spin can ever flip and then flip back. Edge
    colors (the couplings) stay fixed throughout."""

    def construct(self):
        lat_J = _ea_couplings()
        edge_to_pair = _sk_edge_assignment()
        sk_J = _sk_couplings_from_lattice(edge_to_pair, lat_J)

        sk_pos, edges, circles, arrows, board = _build_ring_board(
            [-1] * SK_N, sk_J,
        )
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.6)

        _, flips = _greedy_descent_ring([-1] * SK_N, sk_J)
        for i in flips:
            self.play(
                Rotate(arrows[i], PI, about_point=sk_pos[i]),
                run_time=FLIP_RUNTIME,
            )
        self.wait(1.6)


class Scene9_SRG_Formation(Scene):
    """Prune K_7 to a sparse random graph (c ≈ 3, 10 edges) and flip any
    still-up spin back to down — simultaneously, in a single play."""

    def construct(self):
        lat_J = _ea_couplings()
        edge_to_pair = _sk_edge_assignment()
        sk_J = _sk_couplings_from_lattice(edge_to_pair, lat_J)
        end_spins = _sk_scene_end_spins(sk_J)   # Scene 8's final configuration

        sk_pos, edges, circles, arrows, board = _build_ring_board(end_spins, sk_J)
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.5)

        kept = _srg_edges()
        removed = [c for c in sk_J if c not in kept]

        flip_down_anims = [
            Rotate(arrows[i], PI, about_point=sk_pos[i])
            for i in range(SK_N) if end_spins[i] == 1
        ]
        prune_anims = [FadeOut(edges[c]) for c in removed]

        self.play(*flip_down_anims, *prune_anims, run_time=1.6)
        self.wait(1.0)


class Scene10_SRG_Evolution(Scene):
    """Zero-T descent on the sparse random graph until stable."""

    def construct(self):
        lat_J = _ea_couplings()
        edge_to_pair = _sk_edge_assignment()
        sk_J = _sk_couplings_from_lattice(edge_to_pair, lat_J)
        kept = _srg_edges()
        srg_J = {c: sk_J[c] for c in kept}

        sk_pos, edges, circles, arrows, board = _build_ring_board(
            [-1] * SK_N, sk_J, kept_pairs=kept,
        )
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.6)

        _, flips = _greedy_descent_ring([-1] * SK_N, srg_J)
        for i in flips:
            self.play(
                Rotate(arrows[i], PI, about_point=sk_pos[i]),
                run_time=FLIP_RUNTIME,
            )
        self.wait(1.5)
