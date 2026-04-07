# tiny filesystem helpers — ensure dirs, dump json (numpy-aware), save npz
import json
from pathlib import Path

import numpy as np

# mkdir -p style — idempotent, returns Path for chaining
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# write obj as json; numpy scalars/arrays go through _json_default
def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)

# save arbitrary named arrays into an .npz archive
def save_npz(path, **arrays):
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(path, **arrays)

# json fallback: numpy arrays -> lists, numpy scalars -> python scalars
def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
