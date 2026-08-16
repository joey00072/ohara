"""A progress dashboard for a running training job.

    python examples/train_status.py --log runs/moe.log --port 8082

Reads the training log directly rather than a tracker, because the log has a line
for *every* iteration while experiment trackers here only receive metrics on the
evaluation interval. That makes the loss curve dense and the time estimate honest
without changing anything about the running job.

Works with the multi-stage ``speedrun.sh`` log and with a single ``train_llama_engine``
or ``train_sft`` log. Pass ``--log`` or let it pick the most recently modified log
in ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30), "IST")

ITER_RE = re.compile(
    r"^iter:\s*(\d+)\s*\|\s*loss:\s*([\d.]+)\s*\|\s*lr:\s*([\d.e+-]+)\s*\|\s*"
    r"time:\s*([\d.]+)s(?:.*?tok/s:\s*([\d,]+))?(?:.*?mfu:\s*([\d.]+)%)?(?:.*?eta:\s*([\d.]+)m)?"
)
VAL_RE = re.compile(
    r"^iter:\s*(\d+)\s*\|\s*val_loss:\s*([\d.]+).*?val_acc:\s*([\d.]+)"
)
VAL_BPB_RE = re.compile(r"val_bpb:\s*([\d.]+)")
STAGE_RE = re.compile(r"^\[(\d)/4\]\s*(.+)$")
BUDGET_RE = re.compile(r"budget:\s*([\d,]+)\s*tokens over\s*([\d,]+)\s*iters")
PARAMS_RE = re.compile(r"params=([\d,]+)")
MAX_ITERS_RE = re.compile(r"max_iters=([\d,]+)")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_log(
    path: Path, history_points: int = 300, total_override: int | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "stage": "waiting",
        "iter": 0,
        "total_iters": None,
        "loss": None,
        "median_step_seconds": None,
        "tokens_per_sec": None,
        "mfu": None,
        "params": None,
        "val": [],
        "history": [],
        "percent": 0.0,
        "remaining_minutes": None,
        "finish_ist": None,
        "now_ist": datetime.now(IST).strftime("%H:%M"),
        "finished": False,
        "error": None,
        "log": str(path),
    }
    if not path.exists():
        state["error"] = f"log not found: {path}"
        return state

    losses: list[tuple[int, float]] = []
    steps: deque[float] = deque(maxlen=200)
    declared_total: int | None = None
    eta_minutes: float | None = None
    # A single-file run has no [n/4] markers, so infer the stage from its output.
    saw_sft_markers = False

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            stage = STAGE_RE.match(line)
            if stage:
                state["stage"] = stage.group(2).strip()
                if stage.group(1) == "4":
                    losses.clear()
                    declared_total = None
                continue

            if "building SFT mixture" in line or "before SFT" in line:
                saw_sft_markers = True
                continue
            if "after SFT" in line or "final val_loss" in line:
                state["finished"] = True
                continue

            budget = BUDGET_RE.search(line)
            if budget:
                declared_total = _int(budget.group(2))
                continue

            horizon = MAX_ITERS_RE.search(line)
            if horizon:
                declared_total = _int(horizon.group(1))
                continue

            params = PARAMS_RE.search(line)
            if params and state["params"] is None:
                state["params"] = _int(params.group(1))
                continue

            validation = VAL_RE.match(line)
            if validation:
                bpb = VAL_BPB_RE.search(line)
                state["val"].append(
                    {
                        "iter": int(validation.group(1)),
                        "loss": float(validation.group(2)),
                        "acc": float(validation.group(3)),
                        "bpb": float(bpb.group(1)) if bpb else None,
                    }
                )
                continue

            step = ITER_RE.match(line)
            if step:
                state["iter"] = int(step.group(1))
                state["loss"] = float(step.group(2))
                steps.append(float(step.group(4)))
                if step.group(5):
                    state["tokens_per_sec"] = _int(step.group(5))
                state["mfu"] = float(step.group(6)) if step.group(6) else None
                eta_minutes = float(step.group(7)) if step.group(7) else eta_minutes
                losses.append((state["iter"], state["loss"]))

    if state["stage"] == "waiting":
        state["stage"] = "supervised finetuning" if saw_sft_markers else "pretraining"

    # Step times are spiky, so pace comes from a median rather than the last step.
    ordered = sorted(steps)
    median = ordered[len(ordered) // 2] if ordered else None
    state["median_step_seconds"] = round(median, 3) if median else None

    # The trainer prints eta = (max_iters - iter) * average_step, so when the log
    # never states the horizon we can recover it from the countdown itself.
    total = total_override or declared_total
    if total is None and eta_minutes is not None and median:
        total = state["iter"] + round(eta_minutes * 60.0 / median)
    state["total_iters"] = total

    if total:
        state["percent"] = round(min(100.0, 100.0 * state["iter"] / total), 2)
        if median and not state["finished"]:
            remaining = max(0, total - state["iter"]) * median
            state["remaining_minutes"] = round(remaining / 60.0, 1)
            state["finish_ist"] = (
                datetime.now(IST) + timedelta(seconds=remaining)
            ).strftime("%H:%M")
    if state["finished"]:
        state["percent"] = 100.0
        state["remaining_minutes"] = 0.0

    if losses:
        stride = max(1, len(losses) // history_points)
        state["history"] = [{"i": i, "l": v} for i, v in losses[::stride]]
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
  body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
  .bar{transition:width .8s cubic-bezier(.4,0,.2,1)}
  .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:rgb(161,161,170)}
  .val{font-size:19px;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
</style>
</head>
<body class="h-full bg-white text-zinc-900 dark:bg-[#0a0a0b] dark:text-zinc-100 antialiased">
<div class="mx-auto max-w-2xl px-6 py-10">

  <div class="mb-8 flex items-center gap-3">
    <div class="h-2 w-2 rounded-full bg-indigo-500"></div>
    <span class="font-semibold tracking-tight">ohara</span>
    <span id="stage" class="rounded-full border border-zinc-200 px-2.5 py-1 text-[11px] font-medium capitalize text-zinc-500 dark:border-white/10 dark:text-zinc-400">–</span>
    <span id="live" class="ml-auto flex items-center gap-1.5 text-[11px] text-zinc-400"></span>
  </div>

  <div class="mb-2 flex items-end gap-3">
    <span id="pct" class="text-7xl font-semibold leading-none tracking-tighter tabular-nums">–</span>
    <span class="pb-1 text-2xl font-medium text-zinc-300 dark:text-zinc-700">%</span>
  </div>
  <div class="mb-3 h-2.5 overflow-hidden rounded-full bg-zinc-100 dark:bg-white/[0.06]">
    <div id="bar" class="bar h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500" style="width:0%"></div>
  </div>
  <div class="mb-10 flex justify-between text-[13px] text-zinc-500 dark:text-zinc-400">
    <span id="iters">–</span>
    <span id="eta" class="font-medium text-zinc-700 dark:text-zinc-200"></span>
  </div>

  <div class="grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-3">
    <div><div class="lbl">Train loss</div><div id="loss" class="val">–</div></div>
    <div><div class="lbl">Val loss</div><div id="vloss" class="val">–</div></div>
    <div><div class="lbl">Val bpb</div><div id="bpb" class="val">–</div></div>
    <div><div class="lbl">Step</div><div id="step" class="val">–</div></div>
    <div><div class="lbl">MFU</div><div id="mfu" class="val">–</div></div>
    <div><div class="lbl">Tokens/s</div><div id="tps" class="val">–</div></div>
  </div>

  <div class="mt-11">
    <div class="lbl mb-3">Training loss <span id="npts" class="normal-case tracking-normal"></span></div>
    <svg id="chart" viewBox="0 0 600 170" preserveAspectRatio="none" class="h-44 w-full">
      <defs>
        <linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient>
        <linearGradient id="f" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="#6366f1" stop-opacity=".18"/><stop offset="1" stop-color="#6366f1" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <path id="fill" fill="url(#f)"/>
      <path id="line" fill="none" stroke="url(#g)" stroke-width="1.8" vector-effect="non-scaling-stroke"/>
    </svg>
    <div class="mt-1 flex justify-between text-[11px] text-zinc-400"><span id="lo">–</span><span id="hi">–</span></div>
  </div>

  <p id="note" class="mt-10 text-[12px] leading-relaxed text-zinc-400 dark:text-zinc-600"></p>
</div>

<script>
const $ = (id) => document.getElementById(id);
const f = (v, d = 2) => (v === null || v === undefined ? "–" : (+v).toFixed(d));

function curve(h) {
  if (!h || h.length < 2) return ["", ""];
  const ls = h.map((p) => p.l);
  const lo = Math.min(...ls), hi = Math.max(...ls), span = hi - lo || 1;
  const pts = h.map((p, i) => {
    const x = (i / (h.length - 1)) * 600;
    const y = 160 - ((p.l - lo) / span) * 150;
    return [x, y];
  });
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return [line, `${line} L600,170 L0,170 Z`, lo, hi];
}

async function tick() {
  try {
    const s = await fetch("/api/status", { cache: "no-store" }).then((r) => r.json());

    $("pct").textContent = f(s.percent, 1);
    $("bar").style.width = `${s.percent}%`;
    $("stage").textContent = s.stage;
    $("iters").textContent = s.total_iters
      ? `iteration ${s.iter.toLocaleString()} of ${s.total_iters.toLocaleString()}`
      : `iteration ${s.iter.toLocaleString()}`;

    $("eta").textContent = s.finished
      ? "finished"
      : s.finish_ist
      ? (s.remaining_minutes > 90
          ? `${(s.remaining_minutes / 60).toFixed(1)} h left · done ~${s.finish_ist} IST`
          : `${Math.round(s.remaining_minutes)} min left · done ~${s.finish_ist} IST`)
      : "";

    $("loss").textContent = f(s.loss, 3);
    $("step").textContent = s.median_step_seconds ? `${f(s.median_step_seconds, 2)}s` : "–";
    $("mfu").textContent = s.mfu === null ? "–" : `${f(s.mfu, 1)}%`;
    $("tps").textContent = s.tokens_per_sec ? s.tokens_per_sec.toLocaleString() : "–";

    const v = s.val && s.val.length ? s.val[s.val.length - 1] : null;
    $("vloss").textContent = v ? f(v.loss, 3) : "–";
    $("bpb").textContent = v && v.bpb !== null ? f(v.bpb, 4) : "–";

    const [line, fill, lo, hi] = curve(s.history);
    $("line").setAttribute("d", line || "");
    $("fill").setAttribute("d", fill || "");
    if (lo !== undefined) { $("lo").textContent = `min ${f(lo, 3)}`; $("hi").textContent = `max ${f(hi, 3)}`; }
    $("npts").textContent = s.history.length ? `· ${s.history.length} points` : "";

    $("live").innerHTML = s.finished
      ? '<span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span> complete'
      : `<span class="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500"></span> ${s.now_ist} IST`;

    $("note").textContent = s.error
      ? s.error
      : `${s.params ? (s.params / 1e6).toFixed(0) + "M params · " : ""}reading ${s.log}`;
  } catch (e) {
    $("live").textContent = "disconnected";
  }
}
tick();
setInterval(tick, 4000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ohara-status"
    protocol_version = "HTTP/1.1"
    log_file: Path
    total_override: int | None

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's required spelling
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = parse_log(self.log_file, total_override=self.total_override)
            self._send(json.dumps(payload).encode("utf-8"), "application/json")
        else:
            self._send(b'{"error":"not found"}', "application/json", status=404)


def newest_log(directory: Path) -> Path | None:
    logs = sorted(directory.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a training progress dashboard")
    parser.add_argument("--log", default=None, help="default: newest *.log under runs/")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--total-iters",
        type=int,
        default=None,
        help="pin the horizon for logs that predate max_iters being printed",
    )
    args = parser.parse_args()

    log = Path(args.log) if args.log else newest_log(Path("runs"))
    if log is None:
        raise SystemExit("no log files found in runs/; pass --log explicitly")
    print(f"reading {log}")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        type("Bound", (Handler,), {"log_file": log, "total_override": args.total_iters}),
    )
    print(f"training status on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
