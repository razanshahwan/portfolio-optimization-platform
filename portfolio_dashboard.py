"""Compatibility entry point for the existing Streamlit deployment.

Both deployment entry points now render the same portfolio builder.
All UI lives in portfolio_platform.py; all research calculations in thesis_engine.py.
"""
from portfolio_platform import *  # noqa: F401,F403
