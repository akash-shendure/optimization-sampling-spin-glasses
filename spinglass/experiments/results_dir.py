"""canonical result-directory layout and loader.

the CLI writes one timestamped directory per run containing panel JSONs
(summary / table / overlap) plus an index.json. this module defines that
layout in one place so downstream notebooks and plotting scripts don't hard-
code filenames, and provides helpers to enumerate and load results."""
import json
from datetime import datetime
from pathlib import Path

from .io import ensure_dir, save_json


PANEL_SUFFIXES = ("summary", "table", "overlap")


def make_run_dir(base, tag):
    """create a timestamped directory beneath base and return its Path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / f"{ts}_{tag}"
    ensure_dir(path)
    return path


def panel_path(run_dir, panel_name, suffix):
    """canonical filename for a panel/suffix pair within a run dir."""
    if suffix not in PANEL_SUFFIXES:
        raise ValueError(f"unknown suffix {suffix!r}; use one of {PANEL_SUFFIXES}")
    return Path(run_dir) / f"{panel_name}_{suffix}.json"


def write_panel(run_dir, panel_name, grouped=None, table=None, overlap=None):
    """write any provided panel components under the canonical filenames."""
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    if grouped is not None:
        save_json(panel_path(run_dir, panel_name, "summary"), grouped)
    if table is not None:
        save_json(panel_path(run_dir, panel_name, "table"), table)
    if overlap is not None:
        save_json(panel_path(run_dir, panel_name, "overlap"), overlap)


def write_index(run_dir, panels):
    """dump an index.json listing the panels written in this run."""
    save_json(Path(run_dir) / "index.json", list(panels))


def list_runs(base):
    """enumerate run directories under base, newest first."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    entries = [p for p in base_path.iterdir() if p.is_dir()]
    entries.sort(key=lambda p: p.name, reverse=True)
    return entries


def latest_run(base, tag_contains=None):
    """the most recent run dir, optionally restricted to names containing tag."""
    for entry in list_runs(base):
        if tag_contains is None or tag_contains in entry.name:
            return entry
    return None


def load_panel(run_dir, panel_name, suffix="summary"):
    """load one panel component; returns None if the file is missing."""
    path = panel_path(run_dir, panel_name, suffix)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_all_panels(run_dir, suffix="summary"):
    """load every panel named in index.json under the given suffix.

    returns a dict keyed by panel name. panels without the requested suffix
    are silently skipped so this works on both mixed and uniform runs."""
    run_dir = Path(run_dir)
    index_path = run_dir / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r") as f:
        index = json.load(f)
    out = {}
    for entry in index:
        name = entry.get("name") if isinstance(entry, dict) else str(entry)
        payload = load_panel(run_dir, name, suffix=suffix)
        if payload is not None:
            out[name] = payload
    return out
