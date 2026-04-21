# walks the viewer from a ferromagnetic 2d lattice through edge coloring,
# greedy relaxation, flip symmetry, ea disorder, and finally rewiring to
# sk and a sparse random graph — building up topological intuition
from pathlib import Path
import colorsys

import numpy as np
from manim import *

# render straight into the scene folder, keep partials in a hidden cache
_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".manim_cache"
config.media_dir = str(_CACHE)
config.video_dir = str(_HERE)
config.partial_movie_dir = str(_CACHE / "partial_movie_files")
config.quality = "high_quality"
config.frame_rate = 50
# white bg so figures match paper
config.background_color = WHITE

# board geometry — 5x6 lattice with tight spacing for a single-screen view
ROWS = 5
COLS = 6
SPACING = 1.05
ARROW_LEN = 0.56
CIRCLE_R = 0.34

# foreground/grid colors; blue=satisfied, red=frustrated by convention
FG = BLACK
GRID = "#555555"
_BLUE_HEX = "#1C758A"
_RED_HEX = "#CF5044"

EQN_SIZE = 44
# nudge board up to leave room for the equation
BOARD_SHIFT = UP * 0.35

# per-beat run times — tuned so the eye can track each visual change
FLIP_RUNTIME = 0.24
DIAGONAL_RUNTIME = 0.32
TREE_LAYER_RUNTIME = 0.32

# slight bias toward +1 so the initial board is visually unbalanced
INIT_SEED = 1
INIT_BIAS = 0.60

# ea disorder: gaussian couplings centered at 0 with unit variance
EA_J0 = 0.0
EA_JV = 1.0
EA_COUPLING_SEED = 17
# tanh scale that maps J to saturation
COUPLING_SCALE = 1.3

# large J used when we just want a fully-saturated color
SAT_J = 8.0

# sk ring parameters — 7 nodes arranged on a circle of radius 2.2
SK_N = 7
SK_RADIUS = 2.2

# sparse random graph: keep 10 of the 21 possible sk edges
SRG_SEED = 123
SRG_K = 10

# lattice site -> world coordinates, centered on origin
def _positions():
    pos = {}
    for i in range(ROWS):
        for j in range(COLS):
            x = (j - (COLS - 1) / 2) * SPACING
            y = ((ROWS - 1) / 2 - i) * SPACING
            pos[(i, j)] = np.array([x, y, 0.0])
    return pos

# canonical ordering of nearest-neighbor edges (right + down)
def _edge_keys():
    keys = []
    for i in range(ROWS):
        for j in range(COLS):
            if j + 1 < COLS:
                keys.append(((i, j), (i, j + 1)))
            if i + 1 < ROWS:
                keys.append(((i, j), (i + 1, j)))
    return keys

# group sites by anti-diagonal i+j — used to animate growth in waves
def _layers():
    groups = {}
    for i in range(ROWS):
        for j in range(COLS):
            groups.setdefault(i + j, []).append((i, j))
    return [groups[k] for k in sorted(groups)]

# up-arrow for +1, down-arrow for -1, centered on the node
def _spin_arrow(center, spin):
    d = UP if spin > 0 else DOWN
    start = center - d * ARROW_LEN / 2
    end = center + d * ARROW_LEN / 2
    return Arrow(
        start, end, buff=0, color=FG, stroke_width=5,
        max_tip_length_to_length_ratio=0.38,
        max_stroke_width_to_length_ratio=14,
    )

# white-filled circle around each site so edges visually terminate at the node
def _node_circle(center):
    return Circle(
        radius=CIRCLE_R, color=FG,
        fill_color=WHITE, fill_opacity=1.0,
        stroke_width=1.8,
    ).move_to(center)

# hex/rgb helpers used to blend coupling colors toward white
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )

# map a coupling value J to a hue-locked color whose saturation tracks |tanh(J)|
def _coupling_color(j_val, scale=COUPLING_SCALE):
    t = float(np.tanh(j_val / scale))
    base_hex = _BLUE_HEX if t >= 0 else _RED_HEX
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    abs_t = abs(t)
    # weak |t| -> near-white, strong -> base
    new_l = 1.0 - abs_t * (1.0 - l)
    new_s = abs_t * s
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s))

# smoothly interpolate a single edge's color between two J values
def _diverging_edge_anim(line, from_j, to_j):
    def update(m, alpha):
        j = from_j + alpha * (to_j - from_j)
        m.set_stroke(color=_coupling_color(j))
    return UpdateFromAlphaFunc(line, update)

# binary edge color: blue when aligned, red when anti-aligned
def _edge_color(s1, s2):
    return _coupling_color(s1 * s2 * SAT_J)

# project a sign +/-1 to a saturated J for the binary case
def _binary_J(sign):
    return float(sign) * SAT_J

# biased random initial spin configuration
def _initial_spins():
    rng = np.random.default_rng(INIT_SEED)
    return rng.choice([-1, 1], size=(ROWS, COLS), p=[1 - INIT_BIAS, INIT_BIAS])

# count satisfied (blue) vs frustrated (red) edges for the caption
def _count_colors(spins):
    blue = 0
    red = 0
    for (a, b) in _edge_keys():
        if spins[a] == spins[b]:
            blue += 1
        else:
            red += 1
    return blue, red

# uniform ferromagnetic couplings — used for the J>0 intro scene
def _ferro_J():
    return {k: 1.0 for k in _edge_keys()}

# steepest-descent: greedily flip the spin with the most-negative dE per step
def _greedy_descent_lattice(initial, J, max_flips=400):
    spins = initial.copy().astype(int)
    flips = []
    while len(flips) < max_flips:
        best = None
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

# edwards-anderson disorder: fresh gaussian per edge
def _ea_couplings():
    rng = np.random.default_rng(EA_COUPLING_SEED)
    return {k: float(rng.normal(EA_J0, EA_JV)) for k in _edge_keys()}

# assemble the lattice as edges+circles+arrows in a single VGroup
def _build_board(spins, colored_edges=False):
    pos = _positions()
    edges = {}
    for k in _edge_keys():
        a, b = k
        pa, pb = pos[a], pos[b]
        unit = (pb - pa) / np.linalg.norm(pb - pa)
        # back off so the line meets the circle edge
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
        # arrows must sit on top of circles
        a_m.set_z_index(2)
        arrows[k] = a_m
    edges_g = VGroup(*edges.values())
    circles_g = VGroup(*circles.values())
    arrows_g = VGroup(*arrows.values())
    board = VGroup(edges_g, circles_g, arrows_g).move_to(ORIGIN).shift(BOARD_SHIFT)
    shifted_pos = {k: p + BOARD_SHIFT for k, p in pos.items()}
    return shifted_pos, edges, circles, arrows, board

# canonical signed-ferromagnet equation pinned to the bottom of the frame
def _eqn_signed():
    return MathTex(
        r"H(s)\,=\,-J",
        r"\sum_{(i,j)\in E} s_i\,s_j",
        r",\;\, J>0",
        color=FG, font_size=EQN_SIZE,
    ).to_edge(DOWN, buff=0.55)

# H written as J * (blue_count - red_count) with the colored counts inlined
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
    # tint counts to match edge palette
    e[2].set_color(_BLUE_HEX)
    e[4].set_color(_RED_HEX)
    return e.to_edge(DOWN, buff=0.55)

# ea form — disordered couplings drawn from N(J0, J^2)
def _eqn_ea():
    return MathTex(
        r"H(s)\,=\,-\sum_{(i,j)\in E} J_{ij}\, s_i\, s_j,\;\;",
        r"J_{ij} \sim \mathcal{N}(J_0, J^2)",
        # slightly smaller to fit two terms
        color=FG, font_size=EQN_SIZE - 4,
    ).to_edge(DOWN, buff=0.55)

# scene 1: introduce the 2d lattice as a graph by growing it diagonal-by-diagonal
class Scene1_Lattice(Scene):

    def construct(self):
        spins = _initial_spins()
        pos, edges, circles, arrows, board = _build_board(spins)
        eqn = _eqn_signed()

        # build each anti-diagonal as a unit: incoming edges, then nodes+arrows
        layers = _layers()
        layer_groups = []
        start_cell = layers[0][0]
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

        # cascade the layers, then drop in the hamiltonian below
        self.play(LaggedStart(*layer_groups, lag_ratio=0.45), run_time=3.0)
        self.play(Write(eqn), run_time=0.9)
        self.wait(1.0)

# scene 2: color every edge by whether it's satisfied (blue) or frustrated (red)
class Scene2_EdgeColoring(Scene):

    def construct(self):
        spins = _initial_spins()
        pos, edges, circles, arrows, board = _build_board(spins)
        eqn_old = _eqn_signed()

        self.add(board, eqn_old)
        self.wait(0.3)

        # cascade-recolor edges, then morph equation into the counting form
        recolor = [
            line.animate.set_stroke(color=_edge_color(spins[a], spins[b]), width=4.5)
            for (a, b), line in edges.items()
        ]
        self.play(LaggedStart(*recolor, lag_ratio=0.035), run_time=1.6)

        blue, red = _count_colors(spins)
        eqn_new = _eqn_count(blue, red)
        self.play(TransformMatchingTex(eqn_old, eqn_new), run_time=1.0)
        self.wait(1.0)

# scene 3: greedy relaxation on the ferro lattice — flips drive red edges to blue
class Scene3_Relaxation(Scene):

    def construct(self):
        spins = _initial_spins().copy()
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.5)

        # play each greedy flip: rotate arrow + recolor only incident edges
        _, flips = _greedy_descent_lattice(spins, _ferro_J())
        current = spins.copy()
        for (i, j) in flips:
            current[i, j] *= -1
            edge_anims = []
            for (a, b) in _edge_keys():
                if (i, j) == a or (i, j) == b:
                    new_sign = int(current[a]) * int(current[b])
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

# scene 4: z2 symmetry — globally flip every spin diagonal-by-diagonal, energy unchanged
class Scene4_FlipSymmetry(Scene):

    def construct(self):
        spins = np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.4)

        # walk through anti-diagonals; flip cells together so the wave is visible
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

# scene 5: introduce ea disorder — couplings become gaussian, edges fade to varied saturation
class Scene5_EA_InitCouplings(Scene):

    def construct(self):
        spins = -np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        blue, red = _count_colors(spins)
        current_eqn = _eqn_count(blue, red)

        self.add(board, current_eqn)
        self.wait(0.4)

        # swap counting equation for the disordered EA form
        ea_eqn = _eqn_ea()
        self.play(
            FadeOut(current_eqn, shift=UP * 0.12),
            FadeIn(ea_eqn, shift=UP * 0.12),
            run_time=0.9,
        )
        self.wait(0.2)

        # animate couplings diverging from saturated to disordered, diagonal by diagonal
        J = _ea_couplings()
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

# scene 6: greedy descent on the ea lattice — frustration means stuck at a worse minimum
class Scene6_EA_Evolution(Scene):

    def construct(self):
        spins = -np.ones((ROWS, COLS), dtype=int)
        pos, edges, circles, arrows, board = _build_board(spins, colored_edges=True)
        J = _ea_couplings()
        for k in _edge_keys():
            edges[k].set_stroke(color=_coupling_color(J[k]))
        eqn = _eqn_ea()

        self.add(board, eqn)
        self.wait(0.4)

        # we don't recolor edges here — just rotate arrows; couplings are fixed by definition
        _, flips = _greedy_descent_lattice(spins, J)
        for (i, j) in flips:
            self.play(
                Rotate(arrows[(i, j)], PI, about_point=pos[(i, j)]),
                run_time=FLIP_RUNTIME,
            )
        self.wait(1.2)

# place sk nodes evenly around a circle, top-most at angle pi/2
def _sk_positions(center=None):
    if center is None:
        center = BOARD_SHIFT
    positions = {}
    for v in range(SK_N):
        angle = 2 * np.pi * v / SK_N + np.pi / 2
        positions[v] = (
            np.array([SK_RADIUS * np.cos(angle), SK_RADIUS * np.sin(angle), 0.0])
            + center
        )
    return positions

# back off the line endpoints by CIRCLE_R so they meet the node disks cleanly
def _sk_line_endpoints(pa, pb):
    d = pb - pa
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return pa.copy(), pa.copy()
    unit = d / L
    return pa + unit * CIRCLE_R, pb - unit * CIRCLE_R

def _world_lattice_positions():
    return {k: p + BOARD_SHIFT for k, p in _positions().items()}

# hungarian-match lattice edges to sk pairs to minimize total morph distance
def _sk_edge_assignment():
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
            # try both orientations and keep the cheaper, recording the swap
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

# choose which 7 lattice sites become the sk nodes (the rest fade out)
def _sk_node_assignment():
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

# many lattice edges map onto the same sk pair — keep one J per canonical pair
def _sk_couplings_from_lattice(edge_to_pair, lat_J):
    survivor_k = {}
    for k in _edge_keys():
        va, vb = edge_to_pair[k]
        if va == vb:
            continue
        canonical = (min(va, vb), max(va, vb))
        # last write wins as the canonical survivor
        survivor_k[canonical] = k
    sk_J = {}
    for k in _edge_keys():
        va, vb = edge_to_pair[k]
        if va == vb:
            continue
        canonical = (min(va, vb), max(va, vb))
        if survivor_k[canonical] == k:
            sk_J[canonical] = lat_J[k]
    return sk_J

# local field on site i within the sk/srg ring
def _local_field_ring(spins, J, i, n):
    h = 0.0
    for j in range(n):
        if j == i:
            continue
        pair = (min(i, j), max(i, j))
        if pair in J:
            h += J[pair] * int(spins[j])
    return h

# steepest-descent for the ring topology — same algorithm as the lattice
def _greedy_descent_ring(initial, J, max_flips=200):
    spins = np.array(initial, dtype=int)
    n = len(spins)
    flips = []
    while len(flips) < max_flips:
        best = None
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

# pick a random k-edge subset of the complete graph for the srg topology
def _srg_edges(seed=SRG_SEED, k=SRG_K):
    all_pairs = [(a, b) for a in range(SK_N) for b in range(a + 1, SK_N)]
    rng = np.random.default_rng(seed)
    rng.shuffle(all_pairs)
    return set(all_pairs[:k])

# slide an arrow from one lattice site to its sk position while rotating spin
def _move_rotate_arrow(arrow, from_pos, to_pos, initial_spin, final_spin):
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

# scene 7: morph the relaxed ea lattice into the fully-connected sk ring
class Scene7_SK_Transition(Scene):

    def construct(self):
        # start from a relaxed configuration so the topology change is the focus
        J = _ea_couplings()
        start_spins = -np.ones((ROWS, COLS), dtype=int)
        final_spins, _ = _greedy_descent_lattice(start_spins, J)

        pos, edges, circles, arrows, board = _build_board(final_spins, colored_edges=True)
        for k in _edge_keys():
            edges[k].set_stroke(color=_coupling_color(J[k]))
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.4)

        # decide where lattice edges/nodes should land in the ring
        sk_pos = _sk_positions()
        edge_to_pair = _sk_edge_assignment()
        sk_to_lat = _sk_node_assignment()
        surviving_lat_keys = set(sk_to_lat.values())
        dying_lat_keys = [k for k in pos if k not in surviving_lat_keys]

        # morph each lattice edge into its target sk-pair segment
        edge_morphs = []
        for k in _edge_keys():
            va, vb = edge_to_pair[k]
            color = _coupling_color(J[k])
            start, end = _sk_line_endpoints(sk_pos[va], sk_pos[vb])
            target = Line(start, end, color=color, stroke_width=4.5)
            target.set_z_index(edges[k].z_index)
            edge_morphs.append(Transform(edges[k], target))

        # slide surviving nodes into ring positions; rotate arrows to the all-down state
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

        # the lattice sites that aren't sk nodes fade away
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

        # after morphing, drop the duplicate/self-loop edges that piled on the same pair
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

# minimal builder for ring-topology scenes (sk + srg)
def _build_ring_board(spins, J_all, kept_pairs=None):
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

# precompute where greedy descent terminates on the sk graph
def _sk_scene_end_spins(sk_J):
    final, _ = _greedy_descent_ring([-1] * SK_N, sk_J)
    return final.tolist()

# scene 8: greedy descent on the dense sk ring
class Scene8_SK_Evolution(Scene):

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

# scene 9: sparsify sk -> sparse random graph by pruning edges and resetting spins
class Scene9_SRG_Formation(Scene):

    def construct(self):
        lat_J = _ea_couplings()
        edge_to_pair = _sk_edge_assignment()
        sk_J = _sk_couplings_from_lattice(edge_to_pair, lat_J)
        end_spins = _sk_scene_end_spins(sk_J)

        sk_pos, edges, circles, arrows, board = _build_ring_board(end_spins, sk_J)
        eqn = _eqn_ea()
        self.add(board, eqn)
        self.wait(0.5)

        # pick the surviving edge subset, fade the rest, and reset all spins down at the same time
        kept = _srg_edges()
        removed = [c for c in sk_J if c not in kept]

        flip_down_anims = [
            Rotate(arrows[i], PI, about_point=sk_pos[i])
            for i in range(SK_N) if end_spins[i] == 1
        ]
        prune_anims = [FadeOut(edges[c]) for c in removed]

        self.play(*flip_down_anims, *prune_anims, run_time=1.6)
        self.wait(1.0)

# scene 10: greedy descent on the sparse random graph
class Scene10_SRG_Evolution(Scene):

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
