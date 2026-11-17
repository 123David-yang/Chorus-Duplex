"""Compatibility entry point for scripts.cli.push_dataset."""

import sys
from pathlib import Path

if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.cli.push_dataset import main

    main()
else:
    from scripts.cli import push_dataset as implementation

    sys.modules[__name__] = implementation
