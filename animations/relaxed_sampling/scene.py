# shows how relaxed sampling lifts discrete spins on the cube corners to
# continuous x in R^N via tanh, then how a regularizer pushes mass outward
# so trajectories stay near {-1,+1}^N — discrete walk, smooth walk, regularized walk
import colorsys
from pathlib import Path

import numpy as np
from manim import *

# render output beside the scene, hide manim's working files in a cache dir
_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".manim_cache"
config.media_dir = str(_CACHE)
config.video_dir = str(_HERE)
config.partial_movie_dir = str(_CACHE / "partial_movie_files")
config.quality = "high_quality"
config.frame_rate = 50
# paper-friendly white background
config.background_color = WHITE

# blue = satisfied / good, red = frustrated / bad — match the spin_configuration scene
FG = BLACK
GRID = "#555555"
_BLUE_HEX = "#236B8E"
_RED_HEX = "#CF5044"
GOOD = _BLUE_HEX
BAD = _RED_HEX

# hex/rgb helpers used to fade edge colors toward white as t approaches 0
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )

# map t in [-1,1] to a hue-locked color; |t|->1 saturates, |t|->0 washes to white
def _color_at_t(t):
    t = max(-1.0, min(1.0, float(t)))
    base_hex = _BLUE_HEX if t >= 0 else _RED_HEX
    h, l, s = colorsys.rgb_to_hls(*_hex_to_rgb(base_hex))
    abs_t = abs(t)
    new_l = 1.0 - abs_t * (1.0 - l)
    new_s = abs_t * s
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s))

# interpolate an edge's stroke color between two t values
def _diverging_edge_anim(line, from_t, to_t):
    def update(m, alpha):
        t = from_t + alpha * (to_t - from_t)
        m.set_stroke(color=_color_at_t(t))
    return UpdateFromAlphaFunc(line, update)

TITLE_SIZE = 40
EQN_SIZE = 40
LABEL_SIZE = 34
SIGN_SIZE = 40

# cube placement on the left half of the frame
CUBE_SCALE = 1.15
CUBE_CENTER = LEFT * 3.0 + DOWN * 0.2
CUBE_EDGE_WIDTH = 4.0
CUBE_DASH_WIDTH = 3.5
# corner tagged HIDDEN sits behind the cube under our projection — drawn dashed
HIDDEN_CORNER = (-1, 1, -1)
NEAR_CORNER = (-1, -1, -1)
FAR_CORNER = (1, 1, 1)

# triangle placement on the right half — explicit example of N=3 spins
TRI_CENTER = RIGHT * 3.2 + DOWN * 0.2
TRI_RADIUS = 1.45
TRI_EDGE_WIDTH = 4.0
TRI_COLORED_WIDTH = 7.0

# axonometric-ish projection: y axis tilts to (0.45, 0.35) for depth
def _project(v):
    x, y, z = v
    return np.array([x + 0.45 * y, z + 0.35 * y, 0.0])

# 8 corners of {-1,+1}^3 projected to 2d screen space
def _cube_vertices():
    coords = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return {c: _project(c) * CUBE_SCALE + CUBE_CENTER for c in coords}

# hypercube neighbors: flip one coordinate at a time
def _neighbors(v):
    out = []
    for axis in range(3):
        n = list(v); n[axis] = -n[axis]
        out.append(tuple(n))
    return out

# edges incident to the hidden corner — drawn dashed for occlusion
def _hidden_edge_keys():
    keys = set()
    for nb in _neighbors(HIDDEN_CORNER):
        keys.add(frozenset((HIDDEN_CORNER, nb)))
    return keys

# enumerate cube edges as undirected pairs, no duplicates
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

# hidden edges become dashed lines, visible ones solid
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

# camera direction used to sort edges by depth so trails land on top correctly
_VIEW_DIR = (-0.391, 0.869, -0.304)

def _vertex_depth(v):
    return _VIEW_DIR[0] * v[0] + _VIEW_DIR[1] * v[1] + _VIEW_DIR[2] * v[2]

def _edge_avg_depth(a, b):
    return 0.5 * (_vertex_depth(a) + _vertex_depth(b))

# split edges into near/middle/far groups so we can animate construction in layers
def _partition_edges(V, depth_aware=False):
    edges = _unique_edges(V.keys())
    hidden = _hidden_edge_keys()
    near_keys = {frozenset((NEAR_CORNER, nb)) for nb in _neighbors(NEAR_CORNER)}
    far_keys = {frozenset((FAR_CORNER, nb)) for nb in _neighbors(FAR_CORNER)}

    near, middle, far = [], [], []
    for (a, b) in edges:
        key = frozenset((a, b))
        mobj = _make_cube_edge(a, b, V, hidden)
        if depth_aware:
            # in walk scenes, raise edges in front so they pass over the particle
            mobj.set_z_index(5 if _edge_avg_depth(a, b) < 0 else 0)
        if key in near_keys:
            near.append(mobj)
        elif key in far_keys:
            far.append(mobj)
        else:
            middle.append(mobj)
    return VGroup(*near), VGroup(*middle), VGroup(*far)

# small black dots marking the 8 corners
def _cube_dots(V):
    return VGroup(*[Dot(V[k], color=FG, radius=0.085).set_z_index(1) for k in V])

# label only the (1,1,1) and (-1,-1,-1) corners — enough to orient the viewer
def _corner_labels(V):
    ppp = MathTex(r"(1,1,1)", color=FG, font_size=LABEL_SIZE).next_to(V[FAR_CORNER], UR, buff=0.14)
    mmm = MathTex(r"(-1,-1,-1)", color=FG, font_size=LABEL_SIZE).next_to(V[NEAR_CORNER], DL, buff=0.14)
    return ppp, mmm

# title that names the space the cube lives in
def _cube_title():
    t = MathTex(r"s \in \{-1,+1\}^3", color=FG, font_size=TITLE_SIZE)
    return t.move_to([CUBE_CENTER[0], 3.1, 0])

# equilateral triangle positions on the right panel (vertices 1,2,3)
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

# antiferromagnetic title — every J_ij = -1, so the triangle is frustrated
def _tri_title():
    t = MathTex(r"J_{ij} = -1", color=FG, font_size=TITLE_SIZE)
    return t.move_to([TRI_CENTER[0], 3.1, 0])

# caption listing the three spin signs (e.g. (-1,+1,-1)) under the triangle
def _caption(signs):
    parts = [r"+1" if s == 1 else r"-1" for s in signs]
    tex = MathTex(
        r"(s_1,s_2,s_3)=(" + parts[0] + r"," + parts[1] + r"," + parts[2] + r")",
        color=FG,
        font_size=EQN_SIZE,
    )
    return tex.move_to([TRI_CENTER[0], -2.9, 0])

# scene 2 path: one edge flip to start the demo
TRIANGLE_DEMO_PATH = [
    (-1, -1, -1),
    (1, -1, -1),
]

# scene 3 path: hamiltonian walk visiting every cube corner exactly once
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

# AFM convention: edge cost is -s_i s_j, so aligned -> +1 (bad), anti -> -1 (good)
def _edge_t(s1, s2):
    return -s1 * s2

def _edge_color(s1, s2):
    return _color_at_t(_edge_t(s1, s2))

# floating +1/-1 sign at a triangle vertex; z-index 4 keeps it above the node
def _one_sign(P, vertex_idx, s):
    return MathTex(
        r"+1" if s == 1 else r"-1",
        color=FG, font_size=SIGN_SIZE,
    ).move_to(P[vertex_idx]).set_z_index(4)

# raise particle above front-facing edges when it sits on a near corner
def _particle_z(vertex):
    return 7 if _vertex_depth(vertex) < 0 else 3

# assemble every shared mobject for scenes 1-3 from a single seed configuration
def _build_scaffold(first_signs, depth_aware=False):
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

    # red highlight blob marking the current cube corner; lifted above near edges
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

# scene 1: introduce the cube on the left, the J=-1 triangle on the right, link them via signs
class Scene1_Cube(Scene):

    def construct(self):
        first = TRIANGLE_DEMO_PATH[0]
        s = _build_scaffold(first)

        # plain (uncolored) triangle edges — used while the example is still abstract
        plain_tri_edges = [
            Line(s["P"][i], s["P"][j], color=FG, stroke_width=TRI_EDGE_WIDTH).set_z_index(1)
            for (i, j) in TRI_EDGE_PAIRS
        ]
        plain_tri_group = VGroup(*plain_tri_edges)

        # grow the cube depth-first: near edges, then middle, then far
        self.play(Create(s["near"], lag_ratio=0.0), run_time=0.55)
        self.play(Create(s["middle"], lag_ratio=0.0), run_time=0.75)
        self.play(Create(s["far"], lag_ratio=0.0), run_time=0.55)

        # add dots, title, and the two labeled corners
        self.play(
            FadeIn(s["dots"]),
            Write(s["cube_title"]),
            FadeIn(s["label_ppp"]),
            FadeIn(s["label_mmm"]),
            run_time=0.9,
        )

        # bring in the right-panel triangle (plain edges + nodes + title)
        self.play(
            Create(plain_tri_group),
            FadeIn(s["tri_nodes"]),
            Write(s["tri_title"]),
            run_time=1.1,
        )

        # tie cube corner to triangle: highlight the corner, color the edges, show signs+caption
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

# scene 2: take one step along the cube — show one spin flips and edges recolor in sync
class Scene2_Traversal(Scene):

    def construct(self):
        first = TRIANGLE_DEMO_PATH[0]
        s = _build_scaffold(first)

        # rebuild the scene 1 end state without re-animating
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

        # walk one cube edge per step: hop highlight, recolor only the affected edges, update sign+caption
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

# colored segment showing the path between two visited corners — z-index puts it in front
def _build_trail(a, b, V):
    trail = Line(V[a], V[b], color=BAD, stroke_width=TRAIL_WIDTH)
    trail.cap_style = CapStyleType.ROUND
    trail.set_z_index(6 if _edge_avg_depth(a, b) < 0 else 2)
    return trail, Create(trail)

# scene 4 trajectory: a noisy random walk through the cube interior in s-space
# values stay in (-1,1) — these are tanh(alpha x) coordinates, not raw x
SCENE4_TRAJ_S = [
    np.array([0.995, 0.995, 0.995]),
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
    np.array([-0.20, -0.05, 0.15]),
]

# screen coords for a continuous s vector — same projection as cube vertices
def _s_to_screen(s):
    return _project(np.array(s, dtype=float)) * CUBE_SCALE + CUBE_CENTER

# nudge a point outward along its radius; emulates the regularizer's effect on s
def _push_outward(s, cap=0.95, times=1):
    p = np.array(s, dtype=float)
    for _ in range(times):
        r2 = float(np.sum(p * p))
        if r2 < 1e-10:
            break
        # gentle reshape; 4th root of r^2/2
        denom = (r2 / 2.0) ** 0.25
        p = np.clip(p / denom, -cap, cap)
    return p

# scene 5 trajectory: replay scene 4 in reverse but pushed outward — regularization in action
SCENE5_TRAJ_S = [SCENE4_TRAJ_S[-1]] + [
    _push_outward(s, times=3) for s in SCENE4_TRAJ_S[-2:2:-1]
]

# directions for the regularizer-pulse arrows: 6 face centers + 8 cube corners (normalized)
REG_SPHERE_DIRS = [
    np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
] + [
    np.array([sx, sy, sz], dtype=float) / np.sqrt(3.0)
    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
]

# one arrow shaft+tip starting at zero length — the updater will stretch it outward later
def _make_pulse_arrow(d):
    center_pos = _s_to_screen(np.zeros(3))
    screen_delta = _s_to_screen(d) - center_pos
    mag = float(np.linalg.norm(screen_delta))
    u = screen_delta / max(mag, 1e-9)
    # in-plane perpendicular for the tip base
    perp = np.array([-u[1], u[0], 0.0])

    # start nearly invisible so the very first frames don't show stray pixels
    eps = 1e-3
    shaft = Line(
        center_pos, center_pos + eps * u,
        color=BLUE_E, stroke_width=7,
    )
    shaft.cap_style = CapStyleType.ROUND
    shaft.set_z_index(8)

    # arrowhead as a custom triangle so we can resize freely with the shaft
    tip = Polygon(
        center_pos + eps * u,
        center_pos + 0.5 * eps * perp,
        center_pos - 0.5 * eps * perp,
        color=BLUE_E, fill_color=BLUE_E, fill_opacity=1.0,
        stroke_width=0,
    )
    tip.set_z_index(8)

    arrow = VGroup(shaft, tip)
    # cached direction (in-screen) for the updater
    arrow.u = u
    arrow.perp = perp
    return arrow

# updater that slides all reg arrows outward as a wave: head leaves origin first, tail follows
def _slide_updater(dirs, r_max=0.9, tail_delay=0.35,
                    tip_len=0.22, tip_width=0.17, min_len_s=0.02):
    inv_remain = 1.0 / (1.0 - tail_delay)
    min_gap = min_len_s / r_max
    def update(group, alpha):
        head_fr = min(alpha * inv_remain, 1.0)
        tail_fr = max(0.0, (alpha - tail_delay) * inv_remain)
        # enforce a minimum visible length so the arrow never collapses to a dot
        if head_fr - tail_fr < min_gap:
            head_fr = tail_fr + min_gap
        for i, d in enumerate(dirs):
            tail_pos = _s_to_screen(tail_fr * r_max * d)
            head_pos = _s_to_screen(head_fr * r_max * d)
            arrow = group[i]
            u = arrow.u
            perp = arrow.perp

            # shrink the tip when the arrow gets short so geometry doesn't invert
            arr_len = float(np.linalg.norm(head_pos - tail_pos))
            eff_tip = min(tip_len, 0.5 * arr_len)
            eff_tip = max(eff_tip, 0.005)
            eff_w = eff_tip * (tip_width / tip_len)

            shaft_end = head_pos - eff_tip * u
            if float(np.linalg.norm(shaft_end - tail_pos)) < 0.002:
                # avoid zero-length shaft
                shaft_end = tail_pos + 0.002 * u
            arrow[0].put_start_and_end_on(tail_pos, shaft_end)

            # rebuild the tip triangle at the new head
            tip_base_ctr = head_pos - eff_tip * u
            arrow[1].set_points_as_corners([
                head_pos,
                tip_base_ctr + (eff_w / 2) * perp,
                tip_base_ctr - (eff_w / 2) * perp,
                head_pos,
            ])
    return update

# scene 3: drop the triangle, traverse every cube corner — emphasize the discrete state space
class Scene3_DiscreteWalk(Scene):

    def construct(self):
        scene2_end = TRIANGLE_DEMO_PATH[-1]
        s = _build_scaffold(scene2_end, depth_aware=True)

        # carry over the scene 2 end state
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

        # swap the per-spin caption for a generic N-dimensional space label
        space_label = MathTex(
            r"\vec{s} \in \{-1, +1\}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.3, 0])
        discrete_label = MathTex(
            r"s_i = \pm 1",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        # tear down the triangle panel and reposition particle to the walk's start
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

        # hop corner-to-corner, leaving a colored trail behind each step
        for a, b in zip(DISCRETE_WALK_PATH, DISCRETE_WALK_PATH[1:]):
            particle.set_z_index(7 if _edge_avg_depth(a, b) < 0 else 3)
            _, trail_anim = _build_trail(a, b, V)
            self.play(
                trail_anim,
                particle.animate.move_to(V[b]),
                run_time=0.7,
            )

        self.wait(1.2)

# scene 4: the key relaxation step — replace {-1,+1}^N with s=tanh(alpha x), x in R^N
class Scene4_ContinuousRelaxation(Scene):

    # long enough for the smooth trajectory to read
    RUN_TIME = 7.5

    def construct(self):
        V = _cube_vertices()
        near_edges, middle_edges, far_edges = _partition_edges(V, depth_aware=True)
        dots = _cube_dots(V)
        for d in dots:
            # raise dots above edges so corners stay readable
            d.set_z_index(3)
        label_ppp, label_mmm = _corner_labels(V)

        # reconstruct scene 3 end state: cube + space label + discrete-walk trails
        discrete_space = MathTex(
            r"\vec{s} \in \{-1, +1\}^N",
            color=FG, font_size=EQN_SIZE,
        ).move_to([TRI_CENTER[0], 0.3, 0])
        discrete_values = MathTex(
            r"s_i = \pm 1",
            color=FG, font_size=EQN_SIZE,
        ).move_to([CUBE_CENTER[0], -2.9, 0])

        prior_trails = []
        for a, b in zip(DISCRETE_WALK_PATH, DISCRETE_WALK_PATH[1:]):
            trail, _ = _build_trail(a, b, V)
            prior_trails.append(trail)

        particle = Dot(V[DISCRETE_WALK_PATH[-1]], color=BAD, radius=0.16).set_z_index(3)

        self.add(
            near_edges, middle_edges, far_edges, dots,
            label_ppp, label_mmm,
            discrete_space, discrete_values,
            *prior_trails,
            particle,
        )
        self.wait(0.3)

        # promote the state space to the continuous box [-1,1]^N and introduce x in R^N + tanh map
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

        # erase the discrete trails so the smooth path that follows reads clearly
        self.play(
            *[FadeOut(t) for t in prior_trails],
            Transform(discrete_space, continuous_space),
            FadeIn(x_label),
            FadeOut(discrete_values),
            FadeIn(tanh_label),
            run_time=1.1,
        )
        self.wait(0.35)

        # play the smooth interior trajectory: tracer leaves a continuous red trail
        positions = [_s_to_screen(s) for s in SCENE4_TRAJ_S]

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
        # bezier-smoothed curve through the samples
        path.set_points_smoothly(positions)

        self.play(MoveAlongPath(particle, path), run_time=self.RUN_TIME)

        self.wait(1.5)

# scene 5: add lam * sum(1-s_i^2) regularizer — pulse arrows + an outward-biased trajectory
class Scene5_Regularization(Scene):

    # short visual beat for the regularization "push"
    PULSE_TIME = 3.0
    # longer trajectory so viewer can see it staying outward
    RUN_TIME = 9.0

    def construct(self):
        V = _cube_vertices()
        near_edges, middle_edges, far_edges = _partition_edges(V, depth_aware=True)
        dots = _cube_dots(V)
        for d in dots:
            d.set_z_index(3)
        label_ppp, label_mmm = _corner_labels(V)

        # reproduce scene 4 end state including the smooth trail
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

        particle = Dot(scene4_positions[-1], color=BAD, radius=0.16).set_z_index(3)

        self.add(
            near_edges, middle_edges, far_edges, dots,
            label_ppp, label_mmm,
            continuous_space, x_label, tanh_label,
            scene4_trail, particle,
        )
        self.wait(0.3)

        # introduce the regularized energy: data term in black, penalty in blue
        energy_label = MathTex(
            r"E = -\sum_{ij} J_{ij}\, s_i s_j",
            r"\;+\;",
            r"\lambda \sum_i (1 - s_i^2)",
            color=FG, font_size=EQN_SIZE,
        )
        # tint penalty term to match arrow color
        energy_label[2].set_color(BLUE_E)
        energy_label.move_to([0, -3.1, 0])

        self.play(
            FadeOut(scene4_trail),
            Transform(tanh_label, energy_label),
            run_time=1.0,
        )
        self.wait(0.2)

        # pulse: arrows shoot radially outward from origin, showing the regularizer pushing s away from 0
        pulse_arrows = VGroup(*[_make_pulse_arrow(d) for d in REG_SPHERE_DIRS])
        self.add(pulse_arrows)
        update = _slide_updater(REG_SPHERE_DIRS)

        self.play(
            UpdateFromAlphaFunc(pulse_arrows, update),
            run_time=self.PULSE_TIME,
        )

        self.remove(pulse_arrows)
        self.wait(0.25)

        # second trajectory hugs the cube faces — same dynamics as scene 4 but biased outward
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
