"""Experiment tracking that degrades gracefully when nothing is configured.

``wandb`` is a dependency of this package, so it is always importable — but
importable is not the same as usable. Without an API key it prompts on stdin,
which on a detached training box means the run blocks forever on a question
nobody will answer. That failure is worse than not logging at all.

So the default backend is ``auto``, which resolves in this order:

1. **wandb**, if a key is actually configured (``WANDB_API_KEY`` or a netrc
   entry), or if an explicit offline ``WANDB_MODE`` is selected.
2. **trackio**, which is local-first and needs no account — it writes SQLite
   under ``~/.cache/huggingface/trackio``. https://github.com/gradio-app/trackio
3. **nothing**, with a one-line note saying why, so the run still proceeds.

Loggers here match the interface ``OharaEngine.log_dict`` expects: a
``log_metrics(payload, step=None)`` method.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any, Mapping, Protocol, runtime_checkable


BACKENDS = ("auto", "wandb", "trackio", "none")


@runtime_checkable
class ExperimentLogger(Protocol):
    def log_metrics(self, payload: Mapping[str, Any], step: int | None = None) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    """Accepts metrics and discards them."""

    name = "none"

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def log_metrics(self, payload: Mapping[str, Any], step: int | None = None) -> None:
        return

    def finish(self) -> None:
        return


class _ModuleLogger:
    """Shared adapter over the wandb-style ``init``/``log``/``finish`` API."""

    name = "module"

    def __init__(
        self,
        module: Any,
        *,
        project: str,
        run_name: str | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        self.module = module
        kwargs: dict[str, Any] = {"project": project}
        if run_name:
            kwargs["name"] = run_name
        if config:
            kwargs["config"] = dict(config)
        self.run = module.init(**kwargs)

    def log_metrics(self, payload: Mapping[str, Any], step: int | None = None) -> None:
        # Both backends accept a step kwarg; passing None lets them auto-increment.
        if step is None:
            self.module.log(dict(payload))
        else:
            self.module.log(dict(payload), step=step)

    def finish(self) -> None:
        try:
            self.module.finish()
        except Exception:
            # A tracker failing to close must not take the training run with it.
            pass


class WandbLogger(_ModuleLogger):
    name = "wandb"


class TrackioLogger(_ModuleLogger):
    name = "trackio"

    # Trackio owns these column names. Rename them before logging instead of
    # letting Trackio hide them behind its opaque ``__<name>`` fallback. The
    # trainer's ``time`` value is the duration of one optimization step, not a
    # wall-clock timestamp, hence the more precise public name below.
    _reserved_metric_names = {
        "metrics": "logged_metrics",
        "project": "logged_project",
        "run": "logged_run",
        "step": "logged_step",
        "time": "step_time_s",
        "timestamp": "logged_timestamp",
    }

    def __init__(
        self,
        module: Any,
        *,
        project: str,
        run_name: str | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(
            module,
            project=project,
            run_name=run_name,
            config=config,
        )
        self._last_explicit_step: int | None = None
        self._metric_names_at_step: set[str] = set()
        self._finished = False

    def log_metrics(self, payload: Mapping[str, Any], step: int | None = None) -> None:
        metrics = {
            self._reserved_metric_names.get(name, name): value
            for name, value in payload.items()
        }

        # Trainer logs the optimization metrics and evaluation metrics in two
        # calls at evaluation steps. Trackio stores both calls as independent
        # rows, unlike W&B's same-step merge, so repeated fields otherwise show
        # duplicate points. Keep every new metric while dropping only fields
        # already written at this exact explicit step.
        if step is not None:
            if step != self._last_explicit_step:
                self._last_explicit_step = step
                self._metric_names_at_step.clear()
            metrics = {
                name: value
                for name, value in metrics.items()
                if name not in self._metric_names_at_step
            }
            self._metric_names_at_step.update(metrics)

        if metrics:
            # Use the run returned by init, rather than Trackio's module-global
            # current run. This keeps logs attached to the intended run and is
            # safe if another library initializes Trackio in the same process.
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            # module.finish() also clears Trackio's context variable, which
            # prevents its atexit hook from flushing the same run a second time.
            # Only use it when the module-global run is still ours; otherwise
            # close our exact run without touching another caller's session.
            context_vars = getattr(self.module, "context_vars", None)
            current_run = getattr(context_vars, "current_run", None)
            if current_run is not None and current_run.get() is self.run:
                self.module.finish()
            else:
                self.run.finish()
        except Exception:
            # A tracker failing to flush must not take the training run with it.
            pass


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def wandb_is_configured() -> bool:
    """Whether wandb can log without prompting for credentials.

    An explicit offline mode counts because it logs locally without a key.
    Disabled mode deliberately does not: in ``auto`` it means W&B is unusable,
    so Trackio should receive the metrics instead of a no-op W&B run.
    """
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in {"offline", "dryrun"}:
        return True
    if mode == "disabled":
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb

        # Reads ~/.netrc and any cached login without starting a session.
        return bool(wandb.api.api_key)
    except Exception:
        return False


def create_logger(
    backend: str = "auto",
    *,
    project: str = "ohara",
    run_name: str | None = None,
    config: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> ExperimentLogger:
    """Build a metrics logger, falling back when a backend is unusable.

    Never raises for ``auto``: a training run should not die because tracking is
    unavailable. Explicitly naming a backend that cannot be used *does* raise,
    since that request cannot be honoured silently.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    def announce(message: str) -> None:
        if verbose:
            print(message)

    if backend == "none":
        return NullLogger("disabled")

    if backend == "wandb":
        if not _module_available("wandb"):
            raise RuntimeError("wandb is not installed")
        if not wandb_is_configured():
            raise RuntimeError(
                "wandb has no API key configured. Run `wandb login`, set WANDB_API_KEY, "
                "or set WANDB_MODE=offline. Use --logger trackio for keyless tracking."
            )
        announce(f"tracking with wandb (project={project})")
        return WandbLogger(
            importlib.import_module("wandb"),
            project=project,
            run_name=run_name,
            config=config,
        )

    if backend == "trackio":
        if not _module_available("trackio"):
            raise RuntimeError("trackio is not installed. Run `uv sync` to install it")
        announce(f"tracking with trackio (project={project}, local)")
        return TrackioLogger(
            importlib.import_module("trackio"),
            project=project,
            run_name=run_name,
            config=config,
        )

    # backend == "auto"
    if _module_available("wandb") and wandb_is_configured():
        try:
            logger = WandbLogger(
                importlib.import_module("wandb"),
                project=project,
                run_name=run_name,
                config=config,
            )
            announce(f"tracking with wandb (project={project})")
            return logger
        except Exception as error:  # noqa: BLE001 - fall through to the next backend
            announce(f"wandb unavailable ({type(error).__name__}: {error}); trying trackio")

    if _module_available("trackio"):
        try:
            logger = TrackioLogger(
                importlib.import_module("trackio"),
                project=project,
                run_name=run_name,
                config=config,
            )
            announce(f"tracking with trackio (project={project}, local, no account needed)")
            return logger
        except Exception as error:  # noqa: BLE001 - fall through to the null logger
            announce(f"trackio unavailable ({type(error).__name__}: {error}); metrics not tracked")
            return NullLogger(str(error))

    announce(
        "no experiment tracking: wandb has no API key and trackio is not installed "
        "(`uv sync` installs the local tracker). Training will still print metrics."
    )
    return NullLogger("no usable backend")
