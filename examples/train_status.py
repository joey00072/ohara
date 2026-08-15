"""A small progress dashboard for a running speedrun.

    python examples/train_status.py --log runs/speedrun.log --port 8082

Tails the pipeline log and serves a page showing how far along the run is,
what it is currently doing, and the loss curve so far. Read-only: it never
touches the training job, so it is safe to start and stop at any time.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# "iter: 1742 | loss: 3.1137 | lr: 1.2e-02 | time: 4.0s | tok/s: 131,643 | mfu: 20.4% | eta: 88.3m"
ITER_RE = re.compile(
    r"^iter:\s*(\d+)\s*\|\s*loss:\s*([\d.]+)\s*\|\s*lr:\s*([\d.e+-]+)\s*\|\s*"
    r"time:\s*([\d.]+)s(?:.*?tok/s:\s*([\d,]+))?(?:.*?mfu:\s*([\d.]+)%)?(?:.*?eta:\s*([\d.]+)m)?"
)
VAL_RE = re.compile(r"^iter:\s*(\d+)\s*\|\s*val_loss:\s*([\d.]+).*?val_bpb:\s*([\d.]+)")
STAGE_RE = re.compile(r"^\[(\d)/4\]\s*(.+)$")
BUDGET_RE = re.compile(r"budget:\s*([\d,]+)\s*tokens over\s*([\d,]+)\s*iters")
MODEL_RE = re.compile(r"model:\s*(\d+)\s*effective params, hidden=(\d+)")

STAGE_NAMES = {
    1: "staging corpus",
    2: "planning",
    3: "pretraining",
    4: "supervised finetuning",
}


def parse_log(path: Path, sft_iters: int, history_points: int = 240) -> dict[str, Any]:
    """Extract run state from the pipeline log.

    Loss history is decimated to at most ``history_points`` samples so the page
    stays small no matter how long the run gets.
    """
    state: dict[str, Any] = {
        "stage": 0,
        "stage_name": "waiting",
        "iter": 0,
        "total_iters": None,
        "loss": None,
        "step_seconds": None,
        "tokens_per_sec": None,
        "mfu": None,
        "eta_minutes": None,
        "val": [],
        "history": [],
        "params": None,
        "hidden": None,
        "serving": False,
        "error": None,
    }
    if not path.exists():
        state["error"] = f"log not found: {path}"
        return state

    losses: list[tuple[int, float]] = []
    pretrain_iters: int | None = None
    # Step times are spiky (a periodic dataloader stall runs ~4x the median), so
    # the projection uses a trailing median rather than the most recent step.
    recent_steps: deque[float] = deque(maxlen=100)

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            stage = STAGE_RE.match(line)
            if stage:
                index = int(stage.group(1))
                state["stage"] = index
                state["stage_name"] = STAGE_NAMES.get(index, stage.group(2))
                if index == 4:
                    # SFT restarts the iteration counter from zero.
                    state["iter"] = 0
                    losses.clear()
                continue

            budget = BUDGET_RE.search(line)
            if budget:
                pretrain_iters = int(budget.group(2).replace(",", ""))
                continue

            model = MODEL_RE.search(line)
            if model:
                state["params"] = int(model.group(1))
                state["hidden"] = int(model.group(2))
                continue

            if "chat UI on" in line or "ohara chat UI" in line:
                state["serving"] = True
                continue

            validation = VAL_RE.match(line)
            if validation:
                state["val"].append(
                    {
                        "iter": int(validation.group(1)),
                        "loss": float(validation.group(2)),
                        "bpb": float(validation.group(3)),
                    }
                )
                continue

            step = ITER_RE.match(line)
            if step:
                state["iter"] = int(step.group(1))
                state["loss"] = float(step.group(2))
                state["step_seconds"] = float(step.group(4))
                recent_steps.append(state["step_seconds"])
                if step.group(5):
                    state["tokens_per_sec"] = int(step.group(5).replace(",", ""))
                state["mfu"] = float(step.group(6)) if step.group(6) else None
                state["eta_minutes"] = float(step.group(7)) if step.group(7) else None
                losses.append((state["iter"], state["loss"]))

    state["total_iters"] = sft_iters if state["stage"] >= 4 else pretrain_iters
    state["pretrain_iters"] = pretrain_iters
    state["sft_iters"] = sft_iters

    if losses:
        stride = max(1, len(losses) // history_points)
        state["history"] = [
            {"iter": i, "loss": value} for i, value in losses[::stride]
        ]

    # Overall progress weights the two training stages by their iteration counts,
    # so the bar does not jump back to zero when SFT starts.
    total_work = (pretrain_iters or 0) + sft_iters
    if total_work:
        done = (
            (pretrain_iters or 0) + state["iter"]
            if state["stage"] >= 4
            else min(state["iter"], pretrain_iters or state["iter"])
        )
        state["overall_percent"] = round(100.0 * done / total_work, 2)
    else:
        state["overall_percent"] = 0.0

    if state["total_iters"]:
        state["stage_percent"] = round(100.0 * state["iter"] / state["total_iters"], 2)
    else:
        state["stage_percent"] = 0.0

    # Remaining time covers the stages still to run, not just the current one.
    ordered = sorted(recent_steps)
    step_seconds = ordered[len(ordered) // 2] if ordered else None
    state["median_step_seconds"] = round(step_seconds, 3) if step_seconds else None
    if step_seconds and pretrain_iters:
        if state["stage"] >= 4:
            remaining = max(0, sft_iters - state["iter"])
        else:
            remaining = max(0, pretrain_iters - state["iter"]) + sft_iters
        state["remaining_minutes"] = round(remaining * step_seconds / 60.0, 1)
    else:
        state["remaining_minutes"] = None
    return state


PAGE = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ohara · training</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config={darkMode:"media"}</script>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='12' fill='%236366f1'/></svg>">
<style>
  body{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
  .tick{transition:width .6s cubic-bezier(.4,0,.2,1)}
</style>
</head>
<body class="h-full bg-white text-zinc-900 dark:bg-[#0a0a0b] dark:text-zinc-100 antialiased">
<div class="mx-auto max-w-2xl px-6 py-12">

  <div class="mb-10 flex items-center gap-3">
    <div class="h-2 w-2 rounded-full bg-indigo-500"></div>
    <span class="text-[15px] font-semibold tracking-tight">ohara</span>
    <span id="stage" class="rounded-full border border-zinc-200 px-2.5 py-1 text-[11px] font-medium text-zinc-500 dark:border-white/10 dark:text-zinc-400">–</span>
    <span id="live" class="ml-auto flex items-center gap-1.5 text-[11px] text-zinc-400"></span>
  </div>

  <div class="mb-2 flex items-baseline gap-3">
    <span id="percent" class="text-6xl font-semibold tracking-tighter tabular-nums">–</span>
    <span class="text-2xl font-medium text-zinc-300 dark:text-zinc-700">%</span>
    <span id="eta" class="ml-auto text-sm text-zinc-500 dark:text-zinc-400"></span>
  </div>
  <div class="mb-10 h-2 overflow-hidden rounded-full bg-zinc-100 dark:bg-white/[0.06]">
    <div id="bar" class="tick h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500" style="width:0%"></div>
  </div>

  <div class="grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
    <div><div class="stat-label">Iteration</div><div id="iter" class="stat">–</div></div>
    <div><div class="stat-label">Train loss</div><div id="loss" class="stat">–</div></div>
    <div><div class="stat-label">Val bpb</div><div id="bpb" class="stat">–</div></div>
    <div><div class="stat-label">MFU</div><div id="mfu" class="stat">–</div></div>
    <div><div class="stat-label">Step</div><div id="step" class="stat">–</div></div>
    <div><div class="stat-label">Tokens/s</div><div id="tps" class="stat">–</div></div>
    <div><div class="stat-label">Params</div><div id="params" class="stat">–</div></div>
    <div><div class="stat-label">Stage</div><div id="stagepct" class="stat">–</div></div>
  </div>

  <div class="mt-12">
    <div class="stat-label mb-3">Loss</div>
    <svg id="chart" viewBox="0 0 600 160" preserveAspectRatio="none" class="h-40 w-full overflow-visible">
      <path id="spark" fill="none" stroke="url(#g)" stroke-width="2" vector-effect="non-scaling-stroke"/>
      <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>
    </svg>
  </div>

  <p id="note" class="mt-10 text-[13px] leading-relaxed text-zinc-400 dark:text-zinc-600"></p>
</div>

<style>
  .stat{font-size:20px;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
  .stat-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:rgb(161 161 170)}
</style>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));

function sparkline(history) {
  if (!history || history.length < 2) return "";
  const losses = history.map((p) => p.loss);
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const span = hi - lo || 1;
  return history
    .map((p, i) => {
      const x = (i / (history.length - 1)) * 600;
      const y = 150 - ((p.loss - lo) / span) * 140;
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

async function refresh() {
  try {
    const s = await fetch("/api/status", { cache: "no-store" }).then((r) => r.json());

    $("percent").textContent = fmt(s.overall_percent, 1);
    $("bar").style.width = `${s.overall_percent}%`;
    $("stage").textContent = s.stage_name;
    $("iter").textContent = s.total_iters
      ? `${s.iter.toLocaleString()} / ${s.total_iters.toLocaleString()}`
      : s.iter.toLocaleString();
    $("loss").textContent = fmt(s.loss, 3);
    $("mfu").textContent = s.mfu === null ? "–" : `${fmt(s.mfu, 1)}%`;
    $("step").textContent =
      s.median_step_seconds === null ? "–" : `${fmt(s.median_step_seconds, 2)}s`;
    $("tps").textContent = s.tokens_per_sec ? s.tokens_per_sec.toLocaleString() : "–";
    $("params").textContent = s.params ? `${(s.params / 1e6).toFixed(0)}M` : "–";
    $("stagepct").textContent = `${fmt(s.stage_percent, 1)}%`;

    const lastVal = s.val && s.val.length ? s.val[s.val.length - 1] : null;
    $("bpb").textContent = lastVal ? fmt(lastVal.bpb, 4) : "–";

    $("eta").textContent =
      s.remaining_minutes === null
        ? ""
        : s.remaining_minutes > 90
        ? `~${(s.remaining_minutes / 60).toFixed(1)} h remaining`
        : `~${Math.round(s.remaining_minutes)} min remaining`;

    $("spark").setAttribute("d", sparkline(s.history));

    $("live").innerHTML = s.serving
      ? '<span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span> chat ready'
      : '<span class="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500"></span> running';

    $("note").textContent = s.error
      ? s.error
      : s.serving
      ? "Training finished. The chat model is being served."
      : `Pretraining ${s.pretrain_iters?.toLocaleString() ?? "?"} iters, then ${s.sft_iters.toLocaleString()} iters of SFT.`;
  } catch (error) {
    $("live").textContent = "disconnected";
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "ohara-status"
    protocol_version = "HTTP/1.1"
    log_path: Path
    sft_iters: int

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _respond(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's required spelling
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._respond(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = parse_log(self.log_path, self.sft_iters)
            self._respond(json.dumps(payload).encode("utf-8"), "application/json")
        else:
            self._respond(b'{"error":"not found"}', "application/json", status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a training progress dashboard")
    parser.add_argument("--log", default="runs/speedrun.log")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--sft-iters",
        type=int,
        default=800,
        help="iterations the SFT stage will run, for the overall progress bar",
    )
    args = parser.parse_args()

    handler = type(
        "BoundStatusHandler",
        (StatusHandler,),
        {"log_path": Path(args.log), "sft_iters": args.sft_iters},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"training status on http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
