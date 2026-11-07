"""Compatibility entry point for the original Streamlit deployment.

Streamlit reruns this script after user interactions. Using runpy ensures that
the actual app UI is executed on every rerun, not cached as a Python import.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("portfolio_platform.py")), run_name="__main__")
