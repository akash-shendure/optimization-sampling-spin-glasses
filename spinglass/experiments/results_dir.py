# timestamped run-dir layout and panel json read/write helpers
import json
from datetime import datetime
from pathlib import Path

from .io import ensure_dir, save_json

# each panel can have up to three sibling files distinguished by suffix
PANEL_SUFFIXES = ("summary", "table", "overlap")

# create base/<timestamp>_<tag>/ — sortable lexicographic, unique per second
def make_run_dir(base, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / f"{ts}_{tag}"
    ensure_dir(path)
    return path

# resolve <run_dir>/<panel>_<suffix>.json with suffix validation
def panel_path(run_dir, panel_name, suffix):
    if suffix not in PANEL_SUFFIXES:
        raise ValueError(f"unknown suffix {suffix!r}; use one of {PANEL_SUFFIXES}")
    return Path(run_dir) / f"{panel_name}_{suffix}.json"

# write any combination of the three panel artifacts; None means skip
def write_panel(run_dir, panel_name, grouped=None, table=None, overlap=None):
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    if grouped is not None:
        save_json(panel_path(run_dir, panel_name, "summary"), grouped)
    if table is not None:
        save_json(panel_path(run_dir, panel_name, "table"), table)
    if overlap is not None:
        save_json(panel_path(run_dir, panel_name, "overlap"), overlap)

# index.json lists which panels exist — used by load_all_panels and the CLI
def write_index(run_dir, panels):
    save_json(Path(run_dir) / "index.json", list(panels))

# enumerate run dirs newest-first; empty list if base/ doesn't exist
def list_runs(base):
    base_path = Path(base)
    if not base_path.exists():
        return []
    entries = [p for p in base_path.iterdir() if p.is_dir()]
    # reverse-sort by name == newest first since names start with a timestamp
    entries.sort(key=lambda p: p.name, reverse=True)
    return entries

# most recent run, optionally filtered by substring in the dir name
def latest_run(base, tag_contains=None):
    for entry in list_runs(base):
        if tag_contains is None or tag_contains in entry.name:
            return entry
    return None

# read one panel file; None if missing so callers can treat it as optional
def load_panel(run_dir, panel_name, suffix="summary"):
    path = panel_path(run_dir, panel_name, suffix)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

# load every panel listed in index.json into {name: payload}
def load_all_panels(run_dir, suffix="summary"):
    run_dir = Path(run_dir)
    index_path = run_dir / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r") as f:
        index = json.load(f)
    out = {}
    for entry in index:
        # index entries can be plain names or dicts with a "name" field
        name = entry.get("name") if isinstance(entry, dict) else str(entry)
        payload = load_panel(run_dir, name, suffix=suffix)
        if payload is not None:
            out[name] = payload
    return out
