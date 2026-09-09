"""Exercise the shipped JS using Node's VM and controlled streaming fetches."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_streaming_race_and_code_rendering():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend VM regression tests")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [node, str(root / "tests/webui_frontend.cjs"), str(root / "ohara/webui/static/app.js")],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
