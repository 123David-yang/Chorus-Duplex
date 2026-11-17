"""Compatibility entry point for scripts.cli.run."""

import sys
from pathlib import Path

if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.cli.run import main

    main()
else:
    from scripts.cli import run as implementation

    sys.modules[__name__] = implementation
