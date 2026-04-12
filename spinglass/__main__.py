"""so `python -m spinglass ...` dispatches into the CLI."""
from .cli import main

raise SystemExit(main())
