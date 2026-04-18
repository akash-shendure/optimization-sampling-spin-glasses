# command-line entry: builds models, runs beta-sweep or canonical studies,
# and writes sanitized panels + config to a timestamped run dir
import argparse
import sys

import numpy as np

from .experiments import make_run_dir, save_json, write_index, write_panel
from .experiments.presets import PRESETS, get_preset, list_presets
from .experiments.studies import (
    canonical_study,
    optimization_beta_sweep,
    relaxed_optimization_beta_sweep,
    relaxed_sampling_beta_sweep,
    sampling_beta_sweep,
)
from .models.edwards_anderson import EdwardsAnderson2D
from .models.ising2d import IsingFerromagnet2D
from .models.sk import SherringtonKirkpatrick
from .models.sparse_glass import SparseRandomGlass

# maps cli --model strings to the four supported model classes
MODEL_REGISTRY = {
    "ising2d": IsingFerromagnet2D,
    "ea2d": EdwardsAnderson2D,
    "sparse_glass": SparseRandomGlass,
    "sk": SherringtonKirkpatrick,
}

# parses comma-separated floats; ignores empty fragments from trailing commas
def _parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]

# pack model size args into the {param: [values]} grid form studies expect
def _make_model_kwargs(args):
    if args.model in ("ising2d", "ea2d"):
        return {"L": [int(args.L)]}
    if args.model == "sparse_glass":
        return {"n": [int(args.n)], "c": [float(args.c)]}
    if args.model == "sk":
        return {"n": [int(args.n)]}
    raise ValueError(f"unknown model {args.model}")

# copy parsed args into a json-serializable dict for run provenance
def _config_from_args(args):
    out = {}
    for key, value in vars(args).items():
        if callable(value):  # skip the dispatch func handle
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out

# drop non-jsonable fields from grouped summary rows; arrays -> lists
def _sanitize_summary(grouped):
    out = []
    for row in grouped:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
        out.append(clean)
    return out

# same as _sanitize_summary but also flattens list/tuple cells (trace cols)
def _sanitize_table(table):
    out = []
    for row in table:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                clean[k] = list(v)
        out.append(clean)
    return out

# persist a single panel (grouped/table/optional overlap) under out_dir/name
def _write_panel_to(out_dir, name, panel):
    write_panel(
        out_dir,
        name,
        grouped=_sanitize_summary(panel["grouped"]),
        table=_sanitize_table(panel["table"]),
        overlap=_sanitize_summary(panel["overlap"]) if panel.get("overlap") is not None else None,
    )

# `beta-sweep` subcommand: runs one (task, space) sweep over the beta list
def cmd_beta_sweep(args):
    model_class = MODEL_REGISTRY[args.model]
    betas = _parse_float_list(args.betas)
    model_kwargs = _make_model_kwargs(args)
    out = make_run_dir(args.out, f"beta_sweep_{args.task}_{args.space}_{args.model}")

    # dispatch on (task, space); relaxed-optimization reuses the betas list as lrs
    if args.task == "sampling" and args.space == "discrete":
        panel = sampling_beta_sweep(
            model_class, model_kwargs, betas=betas, n_chains=args.n_chains, n_steps=args.n_steps
        )
    elif args.task == "sampling" and args.space == "relaxed":
        panel = relaxed_sampling_beta_sweep(
            model_class, model_kwargs, betas=betas, n_chains=args.n_chains, n_steps=args.n_steps
        )
    elif args.task == "optimization" and args.space == "discrete":
        panel = optimization_beta_sweep(
            model_class,
            model_kwargs,
            target_betas=betas,
            n_steps=args.n_steps,
            n_restarts=args.n_restarts,
        )
    elif args.task == "optimization" and args.space == "relaxed":
        panel = relaxed_optimization_beta_sweep(
            model_class,
            model_kwargs,
            lrs=betas,  # in relaxed-opt mode --betas carries learning rates
            n_steps=args.n_steps,
            n_restarts=args.n_restarts,
        )
    else:
        raise SystemExit("unsupported (task, space) combination")

    _write_panel_to(out, "beta_sweep", panel)
    save_json(out / "config.json", _config_from_args(args))
    print(f"wrote {out}")
    return 0

# `canonical` subcommand: full discrete+relaxed x sampling+optimization study,
# either ad-hoc from cli args or from a named preset
def cmd_canonical(args):
    # --list-presets short-circuits before any run dir is created
    if getattr(args, "list_presets", False):
        for name in list_presets():
            print(f"  {name:16s} {PRESETS[name]['description']}")
        return 0
    # preset overrides model/betas/budget/etc; otherwise fall back to cli flags
    if getattr(args, "preset", None):
        preset = get_preset(args.preset)
        model_class = preset["model_class"]
        model_kwargs = preset["model_kwargs"]
        betas = preset["betas"]
        n_chains = preset["n_chains"]
        n_restarts = preset["n_restarts"]
        n_disorders = preset["n_disorders"]
        budget = preset["budget"]
        tag = f"canonical_{args.preset}"
    else:
        model_class = MODEL_REGISTRY[args.model]
        model_kwargs = _make_model_kwargs(args)
        betas = _parse_float_list(args.betas)
        n_chains = args.n_chains
        n_restarts = args.n_restarts
        n_disorders = args.n_disorders
        budget = None
        tag = f"canonical_{args.model}"
    out = make_run_dir(args.out, tag)

    panels = canonical_study(
        model_class,
        model_kwargs,
        betas=betas,
        n_steps=args.n_steps,
        n_chains=n_chains,
        n_restarts=n_restarts,
        n_disorders=n_disorders,
        budget=budget,
    )
    # write one file per (space, task) panel and an index linking them
    index = []
    for (space, task), panel in panels.items():
        name = f"{space}_{task}"
        _write_panel_to(out, name, panel)
        index.append({"space": space, "task": task, "name": name})
    write_index(out, index)
    save_json(out / "config.json", _config_from_args(args))
    print(f"wrote {out}")
    return 0

# argparse wiring: shared `common` parent for model/size/budget flags,
# plus per-subcommand options
def build_parser():
    parser = argparse.ArgumentParser(prog="spinglass", description="spinglass benchmark CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", choices=list(MODEL_REGISTRY), default="ising2d")
    common.add_argument("--L", type=int, default=8)
    common.add_argument("--n", type=int, default=64)
    common.add_argument("--c", type=float, default=3.0)
    common.add_argument("--n-steps", dest="n_steps", type=int, default=2000)
    common.add_argument("--n-chains", dest="n_chains", type=int, default=4)
    common.add_argument("--n-restarts", dest="n_restarts", type=int, default=6)
    common.add_argument("--n-disorders", dest="n_disorders", type=int, default=1)
    common.add_argument("--out", default="./results")

    beta = sub.add_parser("beta-sweep", parents=[common])
    beta.add_argument("--task", choices=["sampling", "optimization"], default="sampling")
    beta.add_argument("--space", choices=["discrete", "relaxed"], default="discrete")
    beta.add_argument("--betas", default="0.2,0.5,1.0,2.0")
    beta.set_defaults(func=cmd_beta_sweep)

    canon = sub.add_parser("canonical", parents=[common])
    canon.add_argument("--betas", default="0.2,0.5,1.0,2.0")
    canon.add_argument(
        "--preset",
        choices=list_presets(),
        default=None,
        help="named scaling-study preset; overrides --model/--betas/--n-chains/etc.",
    )
    canon.add_argument(
        "--list-presets",
        dest="list_presets",
        action="store_true",
        help="list available presets and exit",
    )
    canon.set_defaults(func=cmd_canonical)
    return parser

# main entry; argv=None defers to sys.argv
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
