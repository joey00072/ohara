"""Core utility imports must work without the optional Lightning integration."""
import os
import subprocess


def test_utils_and_trainer_import_without_lightning():
    script = '''
import sys
from importlib.abc import MetaPathFinder

class NoLightning(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lightning" or fullname.startswith("lightning."):
            raise ModuleNotFoundError("Lightning deliberately unavailable")

sys.meta_path.insert(0, NoLightning())
import ohara.utils
import ohara.trainer
from examples import core_eval, evaluate_perplexity
assert "ohara.utils.info" not in sys.modules
assert "lightning" not in sys.modules
assert callable(ohara.utils.BetterCycle)
try:
    ohara.utils.model_summary(None)
except ModuleNotFoundError as exc:
    assert "deliberately unavailable" in str(exc)
else:
    raise AssertionError("model_summary must load its optional dependency on demand")
'''
    result = subprocess.run(
        ["uv", "run", "--active", "--no-sync", "python", "-c", script],
        capture_output=True, text=True, timeout=90, env=dict(os.environ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
