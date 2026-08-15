"""Experiment tracking that degrades gracefully when nothing is configured.

``wandb`` is a dependency of this package, so it is always importable — but
importable is not the same as usable. Without an API key it prompts on stdin,
which on a detached training box means the run blocks forever on a question
nobody will answer. That failure is worse than not logging at all.

So the default backend is ``auto``, which resolves in this order:

1. **wandb**, if a key is actually configured (``WANDB_API_KEY``, a netrc entry,
   or an explicit offline/disabled ``WANDB_MODE``).
2. **trackio**, which is local-first and needs no account — it writes SQLite
   under ``~/.cache/huggingface/trackio`` and mirrors ``wandb``'s
   ``init``/``log``/``finish`` API exactly. https://github.com/gradio-app/trackio
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


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def wandb_is_configured() -> bool:
    """Whether wandb can log without prompting for credentials.

    An explicit offline or disabled mode counts: the user has said what they
    want, and neither blocks on input.
    """
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in {"offline", "dryrun", "disabled"}:
        return True
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
            raise RuntimeError("trackio is not installed. Install it with `uv pip install trackio`")
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
        "(`uv pip install trackio` for local tracking). Training will still print metrics."
    )
    return NullLogger("no usable backend")
