"""run the canonical four-panel beta sweep from experiments/studies.py.

this is the single end-to-end driver that the proposal's "first benchmark
study" calls for: a fixed model (or named preset), a temperature sweep,
matched discrete and relaxed optimizer and sampler pairs, and benchmark
tables dumped to disk for downstream plotting. defaults are modest so the
default run finishes in a couple of minutes; pass --preset for scaling
sweeps."""
import argparse
import sys
from pathlib import Path

# let the script run straight from the repo without an install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spinglass.cli import cmd_canonical  # noqa: E402
from spinglass.experiments.presets import list_presets  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="run the canonical spinglass benchmark study")
    parser.add_argument("--model", choices=["ising2d", "ea2d", "sparse_glass", "sk"], default="ea2d")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--c", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--betas", default="0.2,0.5,1.0,2.0,4.0")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=2000)
    parser.add_argument("--n-chains", dest="n_chains", type=int, default=4)
    parser.add_argument("--n-restarts", dest="n_restarts", type=int, default=6)
    parser.add_argument("--n-disorders", dest="n_disorders", type=int, default=1)
    parser.add_argument("--out", default="./results")
    parser.add_argument("--preset", choices=list_presets(), default=None)
    parser.add_argument("--list-presets", dest="list_presets", action="store_true")
    args = parser.parse_args(argv)
    return cmd_canonical(args)


if __name__ == "__main__":
    raise SystemExit(main())
