"""
conftest.py
===========
SIH2026 Root pytest configuration.

Adds person3_dsp/src and person5_acoustic/src to sys.path so that
both packages are importable when running integration tests from the
repository root.

This is necessary because person3_dsp and person5_acoustic each have
their own virtual environment and src-layout; they are NOT installed
in the system-level Python.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# Add P3 and P5 src paths
sys.path.insert(0, str(REPO_ROOT / "person3_dsp" / "src"))
sys.path.insert(0, str(REPO_ROOT / "person5_acoustic" / "src"))
sys.path.insert(0, str(REPO_ROOT))  # for integration/
