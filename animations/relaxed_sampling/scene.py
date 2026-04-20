"""Relaxed-sampling animations for presentation slides.

Each Scene renders to a standalone .mp4 next to this file. Render with:

    manim scene.py Scene1_Cube
    manim scene.py Scene2_Traversal
    manim scene.py Scene3_DiscreteWalk
    manim scene.py Scene4_ContinuousRelaxation
    manim scene.py Scene5_Regularization

Quality and output directory are configured below so the default command
produces 1080p / 50 fps video placed in this folder.
"""
import colorsys
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
GRID = "#555555"
_BLUE_HEX = "#236B8E"  # manim BLUE_E
_RED_HEX = "#CF5044"   # manim RED_E
GOOD = _BLUE_HEX  # satisfied coupling (s_i s_j = -1 under J = -1)
BAD = _RED_HEX    # frustrated coupling (s_i s_j = +1 under J = -1)

# HSL diverging palette on a normalised parameter t in [-1, 1]:
# t = +1 → BLUE_E, t = -1 → RED_E, t = 0 → white. Interpolating t linearly
# during an animation produces a smooth blue↔white↔red sweep across the
# entire run_time, rather than the snap-through-white that a tanh(j)
# mapping would give.


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )


def _color_at_t(t):
    """t in [-1, 1] → hex on the diverging HSL palette through white at 0."""
    t = max(-1.0, min(1.0, float(t)))
    base_hex = _BLUE_HEX if t >= 0 else _RED_HEX
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    abs_t = abs(t)
    new_l = 1.0 - abs_t * (1.0 - l)
    new_s = abs_t * s
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s))


def _diverging_edge_anim(line, from_t, to_t):
    """Animate a line's stroke along the diverging palette by linearly
    interpolating t from from_t to to_t; when the endpoints straddle zero
    the color sweeps smoothly from one saturated endpoint through white to
    the other over the full run_time."""
    def update(m, alpha):
        t = from_t + alpha * (to_t - from_t)
        m.set_stroke(color=_color_at_t(t))
    return UpdateFromAlphaFunc(line, update)

TITLE_SIZE = 40
EQN_SIZE = 40
LABEL_SIZE = 34
SIGN_SIZE = 40

# ---------- cube geometry ----------
CUBE_SCALE = 1.15
CUBE_CENTER = LEFT * 3.0 + DOWN * 0.2
CUBE_EDGE_WIDTH = 4.0
CUBE_DASH_WIDTH = 3.5
HIDDEN_CORNER = (-1, 1, -1)
NEAR_CORNER = (-1, -1, -1)
FAR_CORNER = (1, 1, 1)

# ---------- triangle geometry ----------
TRI_CENTER = RIGHT * 3.2 + DOWN * 0.2
TRI_RADIUS = 1.45
TRI_EDGE_WIDTH = 4.0
TRI_COLORED_WIDTH = 7.0


def _project(v):
    """Oblique projection of a 3D cube coordinate into the 2D manim frame."""
    x, y, z = v
    return np.array([x + 0.45 * y, z + 0.35 * y, 0.0])


def _cube_vertices():
    coords = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return {c: _project(c) * CUBE_SCALE + CUBE_CENTER for c in coords}


def _neighbors(v):
    out = []
    for axis in range(3):
        n = list(v); n[axis] = -n[axis]
        out.append(tuple(n))
    return out


def _hidden_edge_keys():
    keys = set()
    for nb in _neighbors(HIDDEN_CORNER):
        keys.add(frozenset((HIDDEN_CORNER, nb)))
    return keys


def _unique_edges(vertices):
    seen = set()
    out = []
    for v in vertices:
        for nb in _neighbors(v):
            key = frozenset((v, nb))
            if key not in seen:
                seen.add(key)
                out.append((v, nb))
    return out


def _make_cube_edge(a, b, V, hidden_keys):
    if frozenset((a, b)) in hidden_keys:
        edge = DashedLine(
            V[a], V[b], color=FG, stroke_width=CUBE_DASH_WIDTH,
            dash_length=0.12, dashed_ratio=0.55,
        )
    else:
        edge = Line(V[a], V[b], color=FG, stroke_width=CUBE_EDGE_WIDTH)
    edge.set_z_index(0)
    return edge


# Viewing-direction unit vector derived from the oblique projection
# P(x, y, z) = (x + 0.45y, z + 0.35y). The null-space of P (normalised)
# points away from the viewer, so a larger dot product with (x, y, z)
# means the point is farther from the viewer.
_VIEW_DIR = (-0.391, 0.869, -0.304)


def _vertex_depth(v):
    return _VIEW_DIR[0] * v[0] + _VIEW_DIR[1] * v[1] + _VIEW_DIR[2] * v[2]


def _edge_avg_depth(a, b):
    return 0.5 * (_vertex_depth(a) + _vertex_depth(b))


def _partition_edges(V, depth_aware=False):
    """Build the cube edges, partitioned into (near, middle, far) triples
    based on their relation to (-1,-1,-1) / (1,1,1). When `depth_aware` is
    True, each edge receives a z-index derived from its 3D viewing-direction
    depth — edges closer to the viewer (avg depth < 0) sit at z = 5 (above
    the trail) and edges behind the viewer's plane (avg depth > 0) sit at
    z = 0 (behind the trail). This classifies all 12 cube edges including
    the depth edges, not just the front/back faces."""
    edges = _unique_edges(V.keys())
    hidden = _hidden_edge_keys()
    near_keys = {frozenset((NEAR_CORNER, nb)) for nb in _neighbors(NEAR_CORNER)}
    far_keys = {frozenset((FAR_CORNER, nb)) for nb in _neighbors(FAR_CORNER)}

    near, middle, far = [], [], []
    for (a, b) in edges:
        key = frozenset((a, b))
        mobj = _make_cube_edge(a, b, V, hidden)
        if depth_aware:
            mobj.set_z_index(5 if _edge_avg_depth(a, b) < 0 else 0)
        if key in near_keys:
            near.append(mobj)
        elif key in far_keys:
            far.append(mobj)
        else:
            middle.append(mobj)
    return VGroup(*near), VGroup(*middle), VGroup(*far)


def _cube_dots(V):
    return VGroup(*[Dot(V[k], color=FG, radius=0.085).set_z_index(1) for k in V])


def _corner_labels(V):
    ppp = MathTex(r"(1,1,1)", color=FG, font_size=LABEL_SIZE).next_to(V[FAR_CORNER], UR, buff=0.14)
    mmm = MathTex(r"(-1,-1,-1)", color=FG, font_size=LABEL_SIZE).next_to(V[NEAR_CORNER], DL, buff=0.14)
    return ppp, mmm


def _cube_title():
    t = MathTex(r"s \in \{-1,+1\}^3", color=FG, font_size=TITLE_SIZE)
    return t.move_to([CUBE_CENTER[0], 3.1, 0])


def _tri_positions():
    return {
        1: TRI_CENTER + UP * TRI_RADIUS,
        2: TRI_CENTER + DOWN * (TRI_RADIUS * 0.5) + LEFT * (TRI_RADIUS * 0.866),
        3: TRI_CENTER + DOWN * (TRI_RADIUS * 0.5) + RIGHT * (TRI_RADIUS * 0.866),
    }


TRI_EDGE_PAIRS = [(1, 2), (2, 3), (3, 1)]


def _tri_nodes(P):
    return VGroup(*[
        Circle(
            radius=0.42, color=FG, stroke_width=TRI_EDGE_WIDTH,
            fill_color=WHITE, fill_opacity=1,
        ).move_to(P[i]).set_z_index(3)
        for i in (1, 2, 3)
    ])


def _tri_title():
    t = MathTex(r"J_{ij} = -1", color=FG, font_size=TITLE_SIZE)
    return t.move_to([TRI_CENTER[0], 3.1, 0])


def _caption(signs):
    parts = [r"+1" if s == 1 else r"-1" for s in signs]
    tex = MathTex(
        r"(s_1,s_2,s_3)=(" + parts[0] + r"," + parts[1] + r"," + parts[2] + r")",
        color=FG,
        font_size=EQN_SIZE,
    )
    return tex.move_to([TRI_CENTER[0], -2.9, 0])


# Scene2: single edge step from (-1,-1,-1) to the neighbouring vertex, so the
# first change in the triangle configuration (and the resulting frustration
# pattern) can be discussed without the animation progressing further.
TRIANGLE_DEMO_PATH = [
    (-1, -1, -1),
    (1, -1, -1),
]

# Scene3: Hamilton walk visiting all eight vertices via single-spin flips,
# illustrating that the discrete state space is the 3-cube traversed edge-by-edge.
DISCRETE_WALK_PATH = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, 1, 1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
]


def _edge_t(s1, s2):
    """AF triangle: aligned (+1 product) → -1 (red), anti (-1) → +1 (blue)."""
    return -s1 * s2


def _edge_color(s1, s2):
    return _color_at_t(_edge_t(s1, s2))


def _one_sign(P, vertex_idx, s):
    return MathTex(
        r"+1" if s == 1 else r"-1",
        color=FG, font_size=SIGN_SIZE,
    ).move_to(P[vertex_idx]).set_z_index(4)


def _particle_z(vertex):
    """z-index for the cube particle/highlight based on viewing depth:
    7 when the vertex is in front of the viewer's zero plane (on top of
    every cube edge), 3 when behind (below the front-face edges)."""
    return 7 if _vertex_depth(vertex) < 0 else 3


def _build_scaffold(first_signs, depth_aware=False):
    """Build all mobjects for the cube + triangle + first-vertex highlight.

    Returns a dict of references; the triangle edges come pre-colored for
    `first_signs` at full colored width so that Scene2 can add them directly.
    When `depth_aware` is True, the cube edges and the highlight pick up
    depth-based z-indices so the trail/particle respects 3D occlusion.
    """
    V = _cube_vertices()
    near_edges, middle_edges, far_edges = _partition_edges(V, depth_aware=depth_aware)
    dots = _cube_dots(V)
    label_ppp, label_mmm = _corner_labels(V)
    cube_title = _cube_title()

    P = _tri_positions()
    tri_edges = [
        Line(
            P[i], P[j],
            color=_edge_color(first_signs[i - 1], first_signs[j - 1]),
            stroke_width=TRI_COLORED_WIDTH,
        ).set_z_index(1)
        for (i, j) in TRI_EDGE_PAIRS
    ]
    tri_nodes = _tri_nodes(P)
    tri_title = _tri_title()

    highlight_z = _particle_z(first_signs) if depth_aware else 5
    highlight = Dot(V[first_signs], color=BAD, radius=0.16).set_z_index(highlight_z)
    sign_mobs = [_one_sign(P, k + 1, first_signs[k]) for k in range(3)]
    caption = _caption(first_signs)

    return {
        "V": V,
        "P": P,
        "near": near_edges,
        "middle": middle_edges,
        "far": far_edges,
        "dots": dots,
        "label_ppp": label_ppp,
        "label_mmm": label_mmm,
        "cube_title": cube_title,
        "tri_edges": tri_edges,
        "tri_nodes": tri_nodes,
        "tri_title": tri_title,
        "highlight": highlight,
        "sign_mobs": sign_mobs,
        "caption": caption,
    }


class Scene1_Cube(Scene):
    """Build the cube, the triangle scaffold, and land on (-1,-1,-1)."""

    def construct(self):
        first = TRIANGLE_DEMO_PATH[0]
        s = _build_scaffold(first)

        # Triangle edges in Scene1 begin black and recolor when the first
        # vertex is added, so rebuild them at scaffold-width for this scene.
        plain_tri_edges = [
            Line(s["P"][i], s["P"][j], color=FG, stroke_width=TRI_EDGE_WIDTH).set_z_index(1)
            for (i, j) in TRI_EDGE_PAIRS
        ]
        plain_tri_group = VGroup(*plain_tri_edges)

        self.play(Create(s["near"], lag_ratio=0.0), run_time=0.55)
        self.play(Create(s["middle"], lag_ratio=0.0), run_time=0.75)
        self.play(Create(s["far"], lag_ratio=0.0), run_time=0.55)

        self.play(
            FadeIn(s["dots"]),
            Write(s["cube_title"]),
            FadeIn(s["label_ppp"]),
            FadeIn(s["label_mmm"]),
            run_time=0.9,
        )

        self.play(
            Create(plain_tri_group),
            FadeIn(s["tri_nodes"]),
            Write(s["tri_title"]),
            run_time=1.1,
        )

        first_colors = [_edge_color(first[i - 1], first[j - 1]) for (i, j) in TRI_EDGE_PAIRS]
        self.play(
            FadeIn(s["highlight"]),
            *[
                plain_tri_edges[k].animate.set_stroke(
                    color=first_colors[k], width=TRI_COLORED_WIDTH,
                )
                for k in range(3)
            ],
            *[FadeIn(m) for m in s["sign_mobs"]],
            FadeIn(s["caption"]),
            run_time=0.7,
        )
        self.wait(1.0)


class Scene2_Traversal(Scene):
    """Pick up with the (-1,-1,-1) setup in place and walk the front-face path."""

    def construct(self):
        first = TRIANGLE_DEMO_PATH[0]
        s = _build_scaffold(first)

        self.add(
            s["near"], s["middle"], s["far"],
            s["dots"], s["label_ppp"], s["label_mmm"], s["cube_title"],
            *s["tri_edges"], s["tri_nodes"], s["tri_title"],
            s["highlight"], *s["sign_mobs"], s["caption"],
        )
        self.wait(0.3)

        tri_edges = s["tri_edges"]
        sign_mobs = s["sign_mobs"]
        caption = s["caption"]
        highlight = s["highlight"]
        V = s["V"]
        P = s["P"]

        prev = first
        for nxt in TRIANGLE_DEMO_PATH[1:]:
            changed_axis = next(ax for ax in range(3) if prev[ax] != nxt[ax])
            new_sign = _one_sign(P, changed_axis + 1, nxt[changed_axis])
            new_caption = _caption(nxt)

            edge_anims = []
            for k, (i, j) in enumerate(TRI_EDGE_PAIRS):
                old_t = _edge_t(prev[i - 1], prev[j - 1])
                new_t = _edge_t(nxt[i - 1], nxt[j - 1])
                if old_t != new_t:
                    edge_anims.append(_diverging_edge_anim(tri_edges[k], old_t, new_t))

            self.play(
                highlight.animate.move_to(V[nxt]),
                *edge_anims,
                FadeOut(sign_mobs[changed_axis]),
                FadeIn(new_sign),
                FadeOut(caption),
                FadeIn(new_caption),
                run_time=0.75,
            )
            sign_mobs[changed_axis] = new_sign
            caption = new_caption
            prev = nxt
            self.wait(0.5)

        self.wait(1.8)


TRAIL_WIDTH = 6.0


def _build_trail(a, b, V):
    """Solid red trail segment between two cube vertices with depth-aware
    z-index — above the edge the trail lies on, but behind any cube edges
    that are in front of it in 3D."""
    trail = Line(V[a], V[b], color=BAD, stroke_width=TRAIL_WIDTH)
    trail.cap_style = CapStyleType.ROUND
    trail.set_z_index(6 if _edge_avg_depth(a, b) < 0 else 2)
    return trail, Create(trail)


# ---------- continuous-relaxation hand-crafted trajectories ----------
# Plausible-looking motion through [-1, 1]^3; not an actual simulation.

SCENE4_TRAJ_S = [
    np.array([0.995, 0.995, 0.995]),  # ≈ V[(1,1,1)], continues Scene 3 end
    np.array([0.75, 0.85, 0.80]),
    np.array([0.55, 0.60, 0.55]),
    np.array([0.65, 0.30, 0.70]),
    np.array([0.30, 0.55, 0.45]),
    np.array([0.05, 0.20, 0.60]),
    np.array([-0.20, 0.45, 0.30]),
    np.array([-0.05, 0.15, 0.00]),
    np.array([-0.30, -0.10, 0.25]),
    np.array([-0.15, -0.40, 0.45]),
    np.array([0.10, -0.30, 0.15]),
    np.array([0.35, -0.05, -0.20]),
    np.array([0.15, 0.25, -0.40]),
    np.array([-0.10, 0.05, -0.20]),
    np.array([-0.30, 0.30, -0.35]),
    np.array([-0.25, 0.45, -0.05]),
    np.array([0.00, 0.20, 0.10]),
    np.array([-0.20, -0.15, 0.25]),
    np.array([-0.05, 0.10, 0.30]),
    np.array([-0.20, -0.05, 0.15]),  # end
]


def _s_to_screen(s):
    return _project(np.array(s, dtype=float)) * CUBE_SCALE + CUBE_CENTER


def _push_outward(s, cap=0.95, times=1):
    """Radial push p -> p / (|p|²/2)^(1/4) applied `times` times. Each
    application sends r -> sqrt(r) * 2^(1/4), which pushes interior points
    outward. Coordinates are clamped to [-cap, cap] after each application
    so the result always lies in the open cube."""
    p = np.array(s, dtype=float)
    for _ in range(times):
        r2 = float(np.sum(p * p))
        if r2 < 1e-10:
            break
        denom = (r2 / 2.0) ** 0.25
        p = np.clip(p / denom, -cap, cap)
    return p


# Scene 5 wanders the interior the same way Scene 4 does, but every
# coordinate is pushed outward by applying the radial transform three
# times for a stronger push toward the cube surface. We reverse Scene 4
# so the trajectory starts at its final interior point (for continuity)
# and drop the first three near-corner Scene 4 samples so Scene 5 does
# not end up back at V[(1,1,1)].
SCENE5_TRAJ_S = [SCENE4_TRAJ_S[-1]] + [
    _push_outward(s, times=3) for s in SCENE4_TRAJ_S[-2:2:-1]
]


# Sphere-like set of outward directions for the regularisation pulse:
# six face normals plus eight cube-corner diagonals.
REG_SPHERE_DIRS = [
    np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
] + [
    np.array([sx, sy, sz], dtype=float) / np.sqrt(3.0)
    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
]


def _make_pulse_arrow(d):
    """2D arrow built from primitives: a `Line` shaft plus a filled
    `Polygon` triangular tip. Both start as tiny shapes at the cube
    centre; the slide updater drives their positions every frame. Using
    primitives guarantees the triangular tip is always drawn even when
    Manim's `Arrow` would auto-shrink it."""
    center_pos = _s_to_screen(np.zeros(3))
    screen_delta = _s_to_screen(d) - center_pos
    mag = float(np.linalg.norm(screen_delta))
    u = screen_delta / max(mag, 1e-9)
    perp = np.array([-u[1], u[0], 0.0])

    eps = 1e-3
    shaft = Line(
        center_pos, center_pos + eps * u,
        color=BLUE_E, stroke_width=7,
    )
    shaft.cap_style = CapStyleType.ROUND
    shaft.set_z_index(8)

    tip = Polygon(
        center_pos + eps * u,
        center_pos + 0.5 * eps * perp,
        center_pos - 0.5 * eps * perp,
        color=BLUE_E, fill_color=BLUE_E, fill_opacity=1.0,
        stroke_width=0,
    )
    tip.set_z_index(8)

    arrow = VGroup(shaft, tip)
    arrow.u = u
    arrow.perp = perp
    return arrow


def _slide_updater(dirs, r_max=0.9, tail_delay=0.35,
                    tip_len=0.22, tip_width=0.17, min_len_s=0.02):
    """Update that slides each arrow outward through pure motion. Head
    grows from r=0 to r=r_max over the first (1 - tail_delay) fraction of
    α; the tail stays at 0 until α = tail_delay and then grows to r=r_max
    by α = 1. The arrow appears by extending from a point at the centre,
    travels outward with constant length, and disappears by contracting
    to a point at the surface — no opacity fade. Shaft and tip are sized
    each frame so even the tiniest arrow still has a discernible tip."""
    inv_remain = 1.0 / (1.0 - tail_delay)
    min_gap = min_len_s / r_max
    def update(group, alpha):
        head_fr = min(alpha * inv_remain, 1.0)
        tail_fr = max(0.0, (alpha - tail_delay) * inv_remain)
        if head_fr - tail_fr < min_gap:
            head_fr = tail_fr + min_gap
        for i, d in enumerate(dirs):
            tail_pos = _s_to_screen(tail_fr * r_max * d)
            head_pos = _s_to_screen(head_fr * r_max * d)
            arrow = group[i]
            u = arrow.u
            perp = arrow.perp

            arr_len = float(np.linalg.norm(head_pos - tail_pos))
            eff_tip = min(tip_len, 0.5 * arr_len)
            eff_tip = max(eff_tip, 0.005)
            eff_w = eff_tip * (tip_width / tip_len)

            shaft_end = head_pos - eff_tip * u
            if float(np.linalg.norm(shaft_end - tail_pos)) < 0.002:
                shaft_end = tail_pos + 0.002 * u
            arrow[0].put_start_and_end_on(tail_pos, shaft_end)

            tip_base_ctr = head_pos - eff_tip * u
            arrow[1].set_points_as_corners([
                head_pos,
                tip_base_ctr + (eff_w / 2) * perp,
                tip_base_ctr - (eff_w / 2) * perp,
                head_pos,
            ])
    return update


class Scene3_DiscreteWalk(Scene):
    """Continue from Scene2, fade away the triangle panel, and walk the cube."""

    def construct(self):
        # Inherit Scene2's end state: highlight sits at (1,-1,-1), triangle
        # edges already coloured for that vertex, sign labels and caption shown.
        scene2_end = TRIANGLE_DEMO_PATH[-1]
        s = _build_scaffold(scene2_end, depth_aware=True)

        self.add(
            s["near"], s["middle"], s["far"],
            s["dots"], s["label_ppp"], s["label_mmm"], s["cube_title"],
            *s["tri_edges"], s["tri_nodes"], s["tri_title"],
            s["highlight"], *s["sign_mobs"], s["caption"],
        )
        self.wait(0.3)

        V = s["V"]
        particle = s["highlight"]
        particle_start = DISCRETE_WALK_PATH[0]

        space_label = MathTex(
            r"\vec{s} \in \{-1, +1\}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.3, 0])
        discrete_label = MathTex(
            r"s_i = \pm 1",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        self.play(
            FadeOut(VGroup(*s["tri_edges"])),
            FadeOut(s["tri_nodes"]),
            FadeOut(s["tri_title"]),
            *[FadeOut(m) for m in s["sign_mobs"]],
            FadeOut(s["cube_title"]),
            particle.animate.move_to(V[particle_start]),
            Transform(s["caption"], space_label),
            FadeIn(discrete_label),
            run_time=1.1,
        )
        self.wait(0.4)

        for a, b in zip(DISCRETE_WALK_PATH, DISCRETE_WALK_PATH[1:]):
            # Match the particle's z-index to the 3D depth of the edge it
            # is traversing: above the edge when that edge is in front,
            # behind the front-face edges when it is on a back edge.
            particle.set_z_index(7 if _edge_avg_depth(a, b) < 0 else 3)
            _, trail_anim = _build_trail(a, b, V)
            self.play(
                trail_anim,
                particle.animate.move_to(V[b]),
                run_time=0.7,
            )

        self.wait(1.2)


class Scene4_ContinuousRelaxation(Scene):
    """Continue from Scene3, replace the discrete labels with the tanh
    relaxation, and trace a plausible trajectory through the interior of
    [-1, 1]^N starting from the (1,1,1) corner."""

    RUN_TIME = 7.5

    def construct(self):
        # ----- rebuild Scene3's end state -----
        V = _cube_vertices()
        near_edges, middle_edges, far_edges = _partition_edges(V, depth_aware=True)
        dots = _cube_dots(V)
        for d in dots:
            d.set_z_index(3)
        label_ppp, label_mmm = _corner_labels(V)

        discrete_space = MathTex(
            r"\vec{s} \in \{-1, +1\}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.3, 0])
        discrete_values = MathTex(
            r"s_i = \pm 1",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        # The inherited Scene 3 trails carry their Scene-3 depth-aware z-index
        # from _build_trail — segments on front edges sit at z=6 (on top of
        # the front face), segments on back/hidden edges sit at z=2.
        prior_trails = []
        for a, b in zip(DISCRETE_WALK_PATH, DISCRETE_WALK_PATH[1:]):
            trail, _ = _build_trail(a, b, V)
            prior_trails.append(trail)

        # Particle in the interior sits behind the front face (z=5) but
        # above the depth/back edges and the trail.
        particle = Dot(V[DISCRETE_WALK_PATH[-1]], color=BAD, radius=0.16).set_z_index(3)

        self.add(
            near_edges, middle_edges, far_edges, dots,
            label_ppp, label_mmm,
            discrete_space, discrete_values,
            *prior_trails,
            particle,
        )
        self.wait(0.3)

        # ----- transition: fade the discrete trail, rewrite the labels -----
        continuous_space = MathTex(
            r"\vec{s} \in [-1, 1]^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.6, 0])
        x_label = MathTex(
            r"\vec{x} \in \mathbb{R}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], -0.3, 0])
        tanh_label = MathTex(
            r"s_i = \tanh(\alpha x_i),\; x_i \in \mathbb{R}",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        self.play(
            *[FadeOut(t) for t in prior_trails],
            Transform(discrete_space, continuous_space),
            FadeIn(x_label),
            FadeOut(discrete_values),
            FadeIn(tanh_label),
            run_time=1.1,
        )
        self.wait(0.35)

        # ----- plausible continuous trajectory through [-1,1]^3 -----
        positions = [_s_to_screen(s) for s in SCENE4_TRAJ_S]

        # Snap the particle to the trajectory start (~1 px from V[(1,1,1)])
        # before the tracer begins, so the tracer only records the path.
        particle.move_to(positions[0])

        trail_tracer = TracedPath(
            particle.get_center,
            stroke_color=BAD,
            stroke_width=TRAIL_WIDTH,
            cap_style=CapStyleType.ROUND,
        )
        trail_tracer.set_z_index(2)
        self.add(trail_tracer)

        path = VMobject()
        path.set_points_smoothly(positions)

        self.play(MoveAlongPath(particle, path), run_time=self.RUN_TIME)

        self.wait(1.5)


class Scene5_Regularization(Scene):
    """Continue from Scene 4: fade the Scene 4 trail, pulse an outward
    sphere of arrows (the regularisation force), then resume continuous
    motion with an outward drift."""

    PULSE_TIME = 3.0
    RUN_TIME = 9.0

    def construct(self):
        # ----- rebuild Scene 4's end state -----
        V = _cube_vertices()
        near_edges, middle_edges, far_edges = _partition_edges(V, depth_aware=True)
        dots = _cube_dots(V)
        for d in dots:
            d.set_z_index(3)
        label_ppp, label_mmm = _corner_labels(V)

        continuous_space = MathTex(
            r"\vec{s} \in [-1, 1]^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.6, 0])
        x_label = MathTex(
            r"\vec{x} \in \mathbb{R}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], -0.3, 0])
        tanh_label = MathTex(
            r"s_i = \tanh(\alpha x_i),\; x_i \in \mathbb{R}",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        scene4_positions = [_s_to_screen(s) for s in SCENE4_TRAJ_S]
        scene4_trail = VMobject(stroke_color=BAD, stroke_width=TRAIL_WIDTH)
        scene4_trail.set_points_smoothly(scene4_positions)
        scene4_trail.cap_style = CapStyleType.ROUND
        scene4_trail.set_z_index(2)

        # Interior particle: behind front face (z=5) but above trail/back.
        particle = Dot(scene4_positions[-1], color=BAD, radius=0.16).set_z_index(3)

        self.add(
            near_edges, middle_edges, far_edges, dots,
            label_ppp, label_mmm,
            continuous_space, x_label, tanh_label,
            scene4_trail, particle,
        )
        self.wait(0.3)

        # ----- fade out the Scene 4 trail; transform the tanh formula
        #       into the full energy with the regularisation term -----
        energy_label = MathTex(
            r"E = -\sum_{ij} J_{ij}\, s_i s_j",
            r"\;+\;",
            r"\lambda \sum_i (1 - s_i^2)",
            color=FG, font_size=EQN_SIZE,
        )
        energy_label[2].set_color(BLUE_E)
        energy_label.move_to([0, -3.1, 0])

        self.play(
            FadeOut(scene4_trail),
            Transform(tanh_label, energy_label),
            run_time=1.0,
        )
        self.wait(0.2)

        # ----- pulse the outward force-field sphere (2D arrows, sliding) -----
        pulse_arrows = VGroup(*[_make_pulse_arrow(d) for d in REG_SPHERE_DIRS])
        self.add(pulse_arrows)
        update = _slide_updater(REG_SPHERE_DIRS)

        self.play(
            UpdateFromAlphaFunc(pulse_arrows, update),
            run_time=self.PULSE_TIME,
        )

        self.remove(pulse_arrows)
        self.wait(0.25)

        # ----- outward-tending motion through the interior -----
        scene5_positions = [_s_to_screen(s) for s in SCENE5_TRAJ_S]
        particle.move_to(scene5_positions[0])

        tracer5 = TracedPath(
            particle.get_center,
            stroke_color=BAD,
            stroke_width=TRAIL_WIDTH,
            cap_style=CapStyleType.ROUND,
        )
        tracer5.set_z_index(2)
        self.add(tracer5)

        path5 = VMobject()
        path5.set_points_smoothly(scene5_positions)

        self.play(MoveAlongPath(particle, path5), run_time=self.RUN_TIME)

        self.wait(1.5)
