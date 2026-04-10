# zero-dependency driver: discovers test_*.py modules and runs every test_* function
import importlib
import traceback
from pathlib import Path

# walk the tests/ directory and return module stems sorted alphabetically
def discover():
    here = Path(__file__).resolve().parent
    return sorted(p.stem for p in here.glob("test_*.py"))

# import each module, call every test_* callable, collect failures
def run():
    failures = []
    n_run = 0
    for modname in discover():
        module = importlib.import_module(f"tests.{modname}")
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            n_run += 1
            try:
                fn()
                print(f"  ok   {modname}.{name}")
            except Exception as exc:
                # keep going so we see every failure in one pass
                failures.append((modname, name, exc, traceback.format_exc()))
                print(f"  FAIL {modname}.{name}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} / {n_run} tests FAILED")
        for modname, name, _, tb in failures:
            print(f"--- {modname}.{name} ---")
            print(tb)
        # non-zero exit so CI / shell pipelines notice
        raise SystemExit(1)
    print(f"{n_run} / {n_run} tests passed")

if __name__ == "__main__":
    run()
