# AGENTS.md

## Use uv

Run everything through `uv` — never call `pip`, and never invoke a venv's Python
directly (`.venv/bin/python`).

```bash
uv sync                # install/update the environment
uv run python ...      # run a script
uv run pytest          # tests
uv run ruff check .    # lint (experiments/ is excluded on purpose)
```

## ref/ is read-only

`ref/` holds clones of other projects (nanochat, nanomoe) kept for reference. It is
gitignored and not part of this package.

- Read from it freely: use it as reference when implementing or improving something here.
- Never write to it, edit it, or run its scripts. Nothing in `ohara/` may import from it —
  port the idea into our own code and our own style instead.
