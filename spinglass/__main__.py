# enables `python -m spinglass ...`; forwards to cli.main and exits with its code
from .cli import main

raise SystemExit(main())
