#!/usr/bin/env python
"""Run the Signal2Space DL pipeline from the repository root.

    python scripts/run_dl_pipeline.py all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal2space_dl.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
