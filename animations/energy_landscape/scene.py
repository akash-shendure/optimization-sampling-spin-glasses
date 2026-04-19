"""Energy-landscape animations: how geometric frustration produces a
rugged, multi-modal energy landscape.

Two chained scenes on the 3-spin anti-ferromagnetic triangle:

    Scene1_Frustration -- draw the triangle edge-by-edge, fill two of
                          three sites, then place a `?` on the third
                          because no assignment satisfies both remaining
                          bonds.
    Scene2_Alternatives -- pick up from Scene 1: split into two copies
                           side-by-side, one with s3=+1 and one with
                           s3=-1. Every coloured edge is labelled with
                           the bond-energy value (-1 satisfied, +1
                           frustrated).

Render with:

    manim scene.py Scene1_Frustration
    manim scene.py Scene2_Alternatives

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
GOOD = BLUE_E   # satisfied bond (anti-FM: s_i s_j = -1)
BAD = RED_E     # frustrated bond (s_i s_j = +1)

TITLE_SIZE = 44
SIGN_SIZE = 40
QMARK_SIZE = 48
CALC_SIZE = 36

# ---------- triangle geometry ----------
TRI_CENTER = DOWN * 0.2
TRI_RADIUS = 1.45
NODE_RADIUS = 0.42
TRI_EDGE_WIDTH = 4.0
TRI_COLORED_WIDTH = 7.0
CALC_OFFSET = 0.55

# Split offsets for the side-by-side resolution step.
SPLIT_SHIFT = 3.1

TRI_EDGE_PAIRS = [(1, 2), (2, 3), (3, 1)]


def _tri_positions(center=TRI_CENTER):
    return {
        1: center + UP * TRI_RADIUS,
        2: center + DOWN * (TRI_RADIUS * 0.5) + LEFT * (TRI_RADIUS * 0.866),
        3: center + DOWN * (TRI_RADIUS * 0.5) + RIGHT * (TRI_RADIUS * 0.866),
    }


def _tri_nodes(P):
    return VGroup(*[
        Circle(
            radius=NODE_RADIUS, color=FG, stroke_width=TRI_EDGE_WIDTH,
            fill_color=WHITE, fill_opacity=1,
        ).move_to(P[i]).set_z_index(3)
        for i in (1, 2, 3)
    ])


def _plain_edge(P, i, j):
    return Line(P[i], P[j], color=FG, stroke_width=TRI_EDGE_WIDTH).set_z_index(1)


def _edge_color(s_i, s_j):
    """Anti-ferromagnetic: opposite spins satisfy the bond (blue); same spins
    frustrate it (red)."""
    return GOOD if s_i * s_j == -1 else BAD


def _sign_tex(P, idx, s):
    return MathTex(
        r"+1" if s == 1 else r"-1",
        color=FG, font_size=SIGN_SIZE,
    ).move_to(P[idx]).set_z_index(4)


def _calc_label(P, i, j, center, s_i, s_j):
    """Place the bond-energy value (-1 satisfied, +1 frustrated) just outside
    edge (i, j), coloured to match the edge."""
    midpoint = (P[i] + P[j]) / 2
    outward = midpoint - center
    outward = outward / np.linalg.norm(outward)
    prod_str = r"+1" if s_i * s_j == 1 else r"-1"
    return MathTex(
        prod_str,
        color=_edge_color(s_i, s_j), font_size=CALC_SIZE,
    ).move_to(midpoint + outward * CALC_OFFSET)


def _title():
    return Tex(
        r"anti-ferromagnetic triangle $\;(J_{ij}=-1)$",
        color=FG, font_size=TITLE_SIZE,
    ).to_edge(UP, buff=0.55)


def _qmark(P):
    return MathTex(
        "?", color=BAD, font_size=QMARK_SIZE,
    ).move_to(P[3]).set_z_index(4)


class Scene1_Frustration(Scene):
    """Build the triangle edge-by-edge, fill s1 and s2, and end with `?` on
    the third vertex — no single assignment can satisfy both remaining
    bonds."""

    def construct(self):
        P = _tri_positions()
        title = _title()

        edges = {(i, j): _plain_edge(P, i, j) for (i, j) in TRI_EDGE_PAIRS}
        nodes = _tri_nodes(P)

        # Draw the triangle edges one side at a time, then the nodes.
        self.play(
            Write(title),
            LaggedStart(
                *[Create(edges[p]) for p in TRI_EDGE_PAIRS],
                lag_ratio=1.0,
            ),
            run_time=1.8,
        )
        self.play(FadeIn(nodes), run_time=0.45)
        self.wait(0.3)

        # Fill spin 1: +1.
        s1 = 1
        sign1 = _sign_tex(P, 1, s1)
        self.play(FadeIn(sign1), run_time=0.45)
        self.wait(0.2)

        # Fill spin 2: -1. Edge (1,2) becomes blue + energy label -1.
        s2 = -1
        sign2 = _sign_tex(P, 2, s2)
        calc_12 = _calc_label(P, 1, 2, TRI_CENTER, s1, s2)
        self.play(
            FadeIn(sign2),
            edges[(1, 2)].animate.set_stroke(
                color=_edge_color(s1, s2), width=TRI_COLORED_WIDTH,
            ),
            FadeIn(calc_12),
            run_time=0.65,
        )
        self.wait(0.3)

        # `?` on spin 3.
        qmark = _qmark(P)
        self.play(FadeIn(qmark), run_time=0.4)
        self.wait(1.2)


class Scene2_Alternatives(Scene):
    """Pick up from Scene 1: split the triangle into two side-by-side copies,
    one with s3=+1 and one with s3=-1. Each copy ends with exactly one
    frustrated (red, +1) bond."""

    def construct(self):
        P = _tri_positions()
        s1, s2 = 1, -1

        title = _title()
        edges = {(i, j): _plain_edge(P, i, j) for (i, j) in TRI_EDGE_PAIRS}
        edges[(1, 2)].set_stroke(
            color=_edge_color(s1, s2), width=TRI_COLORED_WIDTH,
        )
        nodes = _tri_nodes(P)
        sign1 = _sign_tex(P, 1, s1)
        sign2 = _sign_tex(P, 2, s2)
        calc_12 = _calc_label(P, 1, 2, TRI_CENTER, s1, s2)
        qmark = _qmark(P)

        self.add(title, *edges.values(), nodes, sign1, sign2, calc_12, qmark)
        self.wait(0.5)

        # --- Split into two side-by-side copies ---
        L_shift = LEFT * SPLIT_SHIFT
        R_shift = RIGHT * SPLIT_SHIFT

        L_group = VGroup(
            *edges.values(), nodes, sign1, sign2, qmark, calc_12,
        )

        R_center = TRI_CENTER + R_shift
        PR = _tri_positions(R_center)
        R_edges = {(i, j): _plain_edge(PR, i, j) for (i, j) in TRI_EDGE_PAIRS}
        R_edges[(1, 2)].set_stroke(
            color=_edge_color(s1, s2), width=TRI_COLORED_WIDTH,
        )
        R_nodes = _tri_nodes(PR)
        R_sign1 = _sign_tex(PR, 1, s1)
        R_sign2 = _sign_tex(PR, 2, s2)
        R_qmark = MathTex(
            "?", color=BAD, font_size=QMARK_SIZE,
        ).move_to(PR[3]).set_z_index(4)
        R_calc_12 = _calc_label(PR, 1, 2, R_center, s1, s2)
        R_group = VGroup(
            *R_edges.values(), R_nodes, R_sign1, R_sign2, R_qmark, R_calc_12,
        )

        self.play(
            L_group.animate.shift(L_shift),
            FadeIn(R_group),
            run_time=0.9,
        )
        self.wait(0.3)

        # Resolve: +1 on left node 3, -1 on right node 3, remaining edges
        # coloured with their energy labels. Each triangle ends with exactly
        # one frustrated (red, +1) bond.
        L_center = TRI_CENTER + L_shift
        PL = _tri_positions(L_center)
        L_s3, R_s3 = +1, -1

        L_sign3 = _sign_tex(PL, 3, L_s3)
        R_sign3 = _sign_tex(PR, 3, R_s3)

        L_calc_23 = _calc_label(PL, 2, 3, L_center, s2, L_s3)
        L_calc_31 = _calc_label(PL, 3, 1, L_center, L_s3, s1)
        R_calc_23 = _calc_label(PR, 2, 3, R_center, s2, R_s3)
        R_calc_31 = _calc_label(PR, 3, 1, R_center, R_s3, s1)

        self.play(
            FadeOut(qmark),
            FadeOut(R_qmark),
            FadeIn(L_sign3),
            FadeIn(R_sign3),
            edges[(2, 3)].animate.set_stroke(
                color=_edge_color(s2, L_s3), width=TRI_COLORED_WIDTH,
            ),
            edges[(3, 1)].animate.set_stroke(
                color=_edge_color(L_s3, s1), width=TRI_COLORED_WIDTH,
            ),
            R_edges[(2, 3)].animate.set_stroke(
                color=_edge_color(s2, R_s3), width=TRI_COLORED_WIDTH,
            ),
            R_edges[(3, 1)].animate.set_stroke(
                color=_edge_color(R_s3, s1), width=TRI_COLORED_WIDTH,
            ),
            FadeIn(L_calc_23),
            FadeIn(L_calc_31),
            FadeIn(R_calc_23),
            FadeIn(R_calc_31),
            run_time=0.9,
        )
        self.wait(1.8)
