"""tiny zero-dependency runner so tests are usable without pytest installed.

each test file also has an `if __name__ == '__main__'` block and runs stand-
alone. under pytest, the `test_*` functions are discovered normally. this
script walks the tests/ directory, calls every test function, and prints a
green / red summary."""
import importlib
import traceback
from pathlib import Path


def discover():
    here = Path(__file__).resolve().parent
    return sorted(p.stem for p in here.glob("test_*.py"))


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
            except Exception as exc:  # report and keep going
                failures.append((modname, name, exc, traceback.format_exc()))
                print(f"  FAIL {modname}.{name}: {exc}")
    print()
    if failures:
        print(f"{len(failures)} / {n_run} tests FAILED")
        for modname, name, _, tb in failures:
            print(f"--- {modname}.{name} ---")
            print(tb)
        raise SystemExit(1)
    print(f"{n_run} / {n_run} tests passed")


if __name__ == "__main__":
    run()
