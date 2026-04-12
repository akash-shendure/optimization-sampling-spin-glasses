import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spinglass.cli import cmd_canonical

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
    parser.add_argument("--out", default="./results")
    args = parser.parse_args(argv)
    return cmd_canonical(args)

if __name__ == "__main__":
    raise SystemExit(main())
