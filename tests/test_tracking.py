"""Checks for experiment-tracker selection.

The point of this module is the fallback chain, so these tests pin which backend
gets chosen under each combination of "installed" and "configured", and that
``auto`` never raises — a training run must not die because tracking is
unavailable.
"""

import unittest
from unittest.mock import patch

import pytest

from ohara.tracking import (
    NullLogger,
    TrackioLogger,
    WandbLogger,
    create_logger,
    wandb_is_configured,
)


class FakeRun:
    pass


class FakeTracker:
    """Stands in for the wandb/trackio module surface we use."""

    def __init__(self, fail_on_init=False):
        self.fail_on_init = fail_on_init
        self.init_kwargs = None
        self.logged = []
        self.finished = False

    def init(self, **kwargs):
        if self.fail_on_init:
            raise RuntimeError("no network")
        self.init_kwargs = kwargs
        return FakeRun()

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


def resolve(backend="auto", *, has_wandb=True, wandb_ok=True, has_trackio=True, modules=None):
    """Run create_logger with module availability and wandb keys stubbed out."""
    modules = modules or {}
    available = {"wandb": has_wandb, "trackio": has_trackio}
    with (
        patch("ohara.tracking._module_available", side_effect=lambda n: available.get(n, False)),
        patch("ohara.tracking.wandb_is_configured", return_value=wandb_ok),
        patch("ohara.tracking.importlib.import_module", side_effect=lambda n: modules[n]),
    ):
        return create_logger(backend, project="p", run_name="r", verbose=False)


class AutoResolutionTests(unittest.TestCase):
    def test_prefers_wandb_when_a_key_is_configured(self):
        wandb = FakeTracker()
        logger = resolve(has_wandb=True, wandb_ok=True, modules={"wandb": wandb})
        self.assertIsInstance(logger, WandbLogger)
        self.assertEqual(wandb.init_kwargs["project"], "p")
        self.assertEqual(wandb.init_kwargs["name"], "r")

    def test_falls_back_to_trackio_when_wandb_has_no_key(self):
        trackio = FakeTracker()
        logger = resolve(has_wandb=True, wandb_ok=False, modules={"trackio": trackio})
        self.assertIsInstance(logger, TrackioLogger)
        self.assertIsNotNone(trackio.init_kwargs)

    def test_falls_back_to_trackio_when_wandb_is_not_installed(self):
        trackio = FakeTracker()
        logger = resolve(has_wandb=False, modules={"trackio": trackio})
        self.assertIsInstance(logger, TrackioLogger)

    def test_falls_back_to_trackio_when_wandb_init_fails(self):
        trackio = FakeTracker()
        logger = resolve(
            has_wandb=True,
            wandb_ok=True,
            modules={"wandb": FakeTracker(fail_on_init=True), "trackio": trackio},
        )
        self.assertIsInstance(logger, TrackioLogger)

    def test_falls_back_to_null_when_nothing_is_usable(self):
        logger = resolve(has_wandb=True, wandb_ok=False, has_trackio=False)
        self.assertIsInstance(logger, NullLogger)

    def test_falls_back_to_null_when_trackio_init_fails(self):
        logger = resolve(
            has_wandb=False,
            has_trackio=True,
            modules={"trackio": FakeTracker(fail_on_init=True)},
        )
        self.assertIsInstance(logger, NullLogger)

    def test_auto_never_raises(self):
        # The whole contract: tracking problems must not stop training.
        for has_wandb, wandb_ok, has_trackio in [
            (a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)
        ]:
            modules = {
                "wandb": FakeTracker(fail_on_init=True),
                "trackio": FakeTracker(fail_on_init=True),
            }
            logger = resolve(
                has_wandb=bool(has_wandb),
                wandb_ok=bool(wandb_ok),
                has_trackio=bool(has_trackio),
                modules=modules,
            )
            self.assertTrue(hasattr(logger, "log_metrics"))


class ExplicitBackendTests(unittest.TestCase):
    def test_none_disables_tracking(self):
        self.assertIsInstance(create_logger("none", verbose=False), NullLogger)

    def test_explicit_wandb_without_a_key_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            resolve("wandb", has_wandb=True, wandb_ok=False)
        self.assertIn("API key", str(caught.exception))

    def test_explicit_wandb_when_missing_raises(self):
        with self.assertRaises(RuntimeError):
            resolve("wandb", has_wandb=False)

    def test_explicit_trackio_when_missing_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            resolve("trackio", has_trackio=False)
        self.assertIn("trackio", str(caught.exception))

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_logger("tensorboard", verbose=False)


class LoggerBehaviourTests(unittest.TestCase):
    def test_forwards_metrics_and_step(self):
        module = FakeTracker()
        logger = WandbLogger(module, project="p", run_name=None, config=None)
        logger.log_metrics({"loss": 1.5}, step=7)
        self.assertEqual(module.logged, [({"loss": 1.5}, 7)])

    def test_omits_step_when_none(self):
        module = FakeTracker()
        logger = WandbLogger(module, project="p", run_name=None, config=None)
        logger.log_metrics({"loss": 1.0})
        self.assertEqual(module.logged, [({"loss": 1.0}, None)])

    def test_passes_config_through(self):
        module = FakeTracker()
        WandbLogger(module, project="p", run_name="r", config={"depth": 12})
        self.assertEqual(module.init_kwargs["config"], {"depth": 12})

    def test_finish_survives_a_failing_backend(self):
        class Broken(FakeTracker):
            def finish(self):
                raise RuntimeError("connection reset")

        logger = WandbLogger(Broken(), project="p", run_name=None, config=None)
        logger.finish()  # must not raise

    def test_null_logger_accepts_and_discards(self):
        logger = NullLogger()
        logger.log_metrics({"loss": 1.0}, step=1)
        logger.finish()

    def test_engine_log_dict_contract(self):
        # OharaEngine.log_dict dispatches on this attribute.
        for logger in (NullLogger(), WandbLogger(FakeTracker(), project="p", run_name=None, config=None)):
            self.assertTrue(callable(logger.log_metrics))


class WandbKeyDetectionTests(unittest.TestCase):
    def test_api_key_environment_variable_counts(self):
        with patch.dict("os.environ", {"WANDB_API_KEY": "abc"}, clear=True):
            self.assertTrue(wandb_is_configured())

    def test_offline_modes_count_as_configured(self):
        for mode in ("offline", "dryrun", "disabled"):
            with patch.dict("os.environ", {"WANDB_MODE": mode}, clear=True):
                self.assertTrue(wandb_is_configured(), mode)

    def test_no_key_and_no_netrc_is_unconfigured(self):
        wandb = pytest.importorskip("wandb")
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(wandb.api, "api_key", None):
                self.assertFalse(wandb_is_configured())

    def test_uninstalled_wandb_is_unconfigured(self):
        # The import lives inside the function, so a missing module must be
        # reported as "not configured" rather than raising into the caller.
        import builtins

        real_import = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("no wandb")
            return real_import(name, *args, **kwargs)

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(builtins, "__import__", side_effect=missing):
                self.assertFalse(wandb_is_configured())


if __name__ == "__main__":
    unittest.main()
