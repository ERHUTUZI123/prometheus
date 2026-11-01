"""Approved plotting entry point: import path only; no model or data mutation."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, "/opt/prometheus-v2-python")
runpy.run_path(str(Path(__file__).with_name("plot_results.py")), run_name="__main__")
