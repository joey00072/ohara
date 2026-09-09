"""Side-by-side dashboard for an A/B training comparison.

    python examples/ab_status.py --a runs/xsa_off.log --b runs/xsa_on.log \
        --label-a baseline --label-b "shared-exclusive" --port 8083

Reads two training logs and shows both loss curves plus their difference. The
difference is the point of the page, so it is the thing rendered largest and
coloured: **green when B is better than A** (lower loss), red when it is worse.

Both runs are read from their logs rather than a tracker, so the curves are dense
(one point per iteration) and the page needs nothing from the training jobs.

A note on reading the number: "better" here means lower validation
bits-per-byte, compared at the same iteration. Comparing at the latest iteration
of each run would be meaningless if one is ahead, so the page always aligns on
the newest step both runs have reached.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from train_status import parse_log  # noqa: E402  (same directory)


def align(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare the two runs at the newest iteration both have evaluated."""
    a_val = {entry["iter"]: entry for entry in a["val"]}
    b_val = {entry["iter"]: entry for entry in b["val"]}
    shared = sorted(set(a_val) & set(b_val))
    if not shared:
        return {"iter": None, "pairs": []}
    pairs = [
        {
            "iter": step,
            "a_bpb": a_val[step].get("bpb"),
            "b_bpb": b_val[step].get("bpb"),
            "a_loss": a_val[step]["loss"],
            "b_loss": b_val[step]["loss"],
        }
        for step in shared
    ]
    return {"iter": shared[-1], "pairs": pairs}


def build_payload(
    path_a: Path, path_b: Path, label_a: str, label_b: str, total: int | None
) -> dict[str, Any]:
    a = parse_log(path_a, total_override=total)
    b = parse_log(path_b, total_override=total)
    comparison = align(a, b)

    latest = comparison["pairs"][-1] if comparison["pairs"] else None
    delta_bpb = delta_pct = None
    if latest and latest["a_bpb"] and latest["b_bpb"]:
        delta_bpb = latest["b_bpb"] - latest["a_bpb"]
        delta_pct = 100.0 * delta_bpb / latest["a_bpb"]

    return {
        "a": {"label": label_a, **a},
        "b": {"label": label_b, **b},
        "comparison": comparison,
        # Negative delta means B has the lower bits-per-byte, i.e. B is better.
        "delta_bpb": delta_bpb,
        "delta_pct": delta_pct,
        "b_is_better": None if delta_bpb is None else delta_bpb < 0,
        "now_ist": a["now_ist"],
    }


PAGE = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ohara · A/B</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config={darkMode:"media"}</script>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='12' fill='%236366f1'/></svg>">
<style>
  body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
  .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:rgb(161,161,170)}
  .val{font-size:20px;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
  .bar{transition:width .8s cubic-bezier(.4,0,.2,1)}
  table{font-variant-numeric:tabular-nums}
</style>
</head>
<body class="h-full bg-white text-zinc-900 dark:bg-[#0a0a0b] dark:text-zinc-100 antialiased">
<div class="mx-auto max-w-3xl px-6 py-10">

  <div class="mb-8 flex items-center gap-3">
    <div class="h-2 w-2 rounded-full bg-indigo-500"></div>
    <span class="font-semibold tracking-tight">ohara</span>
    <span class="rounded-full border border-zinc-200 px-2.5 py-1 text-[11px] font-medium text-zinc-500 dark:border-white/10 dark:text-zinc-400">A/B</span>
    <span id="live" class="ml-auto flex items-center gap-1.5 text-[11px] text-zinc-400"></span>
  </div>

  <!-- the headline: the difference -->
  <div id="verdict" class="mb-10 rounded-2xl border p-6">
    <div class="lbl mb-2">Difference in val bits/byte <span id="at-iter" class="normal-case tracking-normal"></span></div>
    <div class="flex items-end gap-4">
      <span id="delta" class="text-6xl font-semibold leading-none tracking-tighter tabular-nums">–</span>
      <span id="delta-pct" class="pb-1 text-2xl font-medium tabular-nums"></span>
    </div>
    <div id="verdict-text" class="mt-3 text-[14px]"></div>
  </div>

  <div class="mb-10 grid gap-6 sm:grid-cols-2">
    <div class="rounded-xl border border-zinc-200 p-5 dark:border-white/10">
      <div class="mb-3 flex items-center gap-2">
        <span class="h-2.5 w-2.5 rounded-sm" style="background:#6366f1"></span>
        <span id="label-a" class="text-[13px] font-medium">A</span>
      </div>
      <div class="grid grid-cols-2 gap-y-3">
        <div><div class="lbl">Val bpb</div><div id="a-bpb" class="val">–</div></div>
        <div><div class="lbl">Train loss</div><div id="a-loss" class="val">–</div></div>
        <div><div class="lbl">Iter</div><div id="a-iter" class="val">–</div></div>
        <div><div class="lbl">Step</div><div id="a-step" class="val">–</div></div>
      </div>
      <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-100 dark:bg-white/[0.06]">
        <div id="a-bar" class="bar h-full rounded-full" style="width:0%;background:#6366f1"></div>
      </div>
    </div>
    <div class="rounded-xl border border-zinc-200 p-5 dark:border-white/10">
      <div class="mb-3 flex items-center gap-2">
        <span class="h-2.5 w-2.5 rounded-sm" style="background:#f59e0b"></span>
        <span id="label-b" class="text-[13px] font-medium">B</span>
      </div>
      <div class="grid grid-cols-2 gap-y-3">
        <div><div class="lbl">Val bpb</div><div id="b-bpb" class="val">–</div></div>
        <div><div class="lbl">Train loss</div><div id="b-loss" class="val">–</div></div>
        <div><div class="lbl">Iter</div><div id="b-iter" class="val">–</div></div>
        <div><div class="lbl">Step</div><div id="b-step" class="val">–</div></div>
      </div>
      <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-100 dark:bg-white/[0.06]">
        <div id="b-bar" class="bar h-full rounded-full" style="width:0%;background:#f59e0b"></div>
      </div>
    </div>
  </div>

  <div class="mb-10">
    <div class="lbl mb-3">Training loss <span id="npts" class="normal-case tracking-normal"></span></div>
    <svg id="chart" viewBox="0 0 640 190" preserveAspectRatio="none" class="h-48 w-full">
      <path id="line-a" fill="none" stroke="#6366f1" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
      <path id="line-b" fill="none" stroke="#f59e0b" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
    </svg>
    <div class="mt-1 flex justify-between text-[11px] text-zinc-400"><span id="lo">–</span><span id="hi">–</span></div>
  </div>

  <div>
    <div class="lbl mb-3">Val bits/byte at each eval</div>
    <table class="w-full text-[13px]">
      <thead class="text-zinc-400">
        <tr class="border-b border-zinc-200 dark:border-white/10">
          <th class="py-2 text-left font-normal">iter</th>
          <th class="py-2 text-right font-normal" id="th-a">A</th>
          <th class="py-2 text-right font-normal" id="th-b">B</th>
          <th class="py-2 text-right font-normal">diff</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>

  <p id="note" class="mt-8 text-[12px] leading-relaxed text-zinc-400 dark:text-zinc-600"></p>
</div>

<script>
const $ = (id) => document.getElementById(id);
const f = (v, d = 4) => (v === null || v === undefined ? "–" : (+v).toFixed(d));

// Lower bits-per-byte is better, so a negative difference is the good case.
const GOOD = { text: "text-emerald-600 dark:text-emerald-400",
               border: "border-emerald-500/40", bg: "bg-emerald-500/5" };
const BAD  = { text: "text-red-600 dark:text-red-400",
               border: "border-red-500/40", bg: "bg-red-500/5" };
const NEUTRAL = { text: "text-zinc-400", border: "border-zinc-200 dark:border-white/10", bg: "" };

function paint(el, theme) {
  el.className = el.className
    .replace(/\\btext-(emerald|red|zinc)-\\d00\\b/g, "")
    .replace(/\\bdark:text-(emerald|red)-\\d00\\b/g, "")
    .trim();
  el.classList.add(...theme.text.split(" "));
}

function curves(a, b) {
  const all = [...a.map(p => p.l), ...b.map(p => p.l)];
  if (all.length < 2) return ["", "", null, null];
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const maxIter = Math.max(...a.map(p => p.i), ...b.map(p => p.i)) || 1;
  const path = (pts) => pts.map((p, i) => {
    const x = (p.i / maxIter) * 640;
    const y = 180 - ((p.l - lo) / span) * 170;
    return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return [path(a), path(b), lo, hi];
}

async function tick() {
  try {
    const s = await fetch("/api/status", { cache: "no-store" }).then(r => r.json());
    const A = s.a, B = s.b;

    $("label-a").textContent = A.label;
    $("label-b").textContent = B.label;
    $("th-a").textContent = A.label;
    $("th-b").textContent = B.label;

    for (const [key, run] of [["a", A], ["b", B]]) {
      const last = run.val.length ? run.val[run.val.length - 1] : null;
      $(`${key}-bpb`).textContent = last && last.bpb !== null ? f(last.bpb) : "–";
      $(`${key}-loss`).textContent = f(run.loss, 3);
      $(`${key}-iter`).textContent = run.total_iters
        ? `${run.iter.toLocaleString()}/${run.total_iters.toLocaleString()}` : run.iter.toLocaleString();
      $(`${key}-step`).textContent = run.median_step_seconds ? `${f(run.median_step_seconds, 2)}s` : "–";
      $(`${key}-bar`).style.width = `${run.percent}%`;
    }

    // headline difference
    const box = $("verdict");
    if (s.delta_bpb === null) {
      $("delta").textContent = "–";
      $("delta-pct").textContent = "";
      $("verdict-text").textContent = "waiting for both runs to reach a shared eval step";
      box.className = `mb-10 rounded-2xl border p-6 ${NEUTRAL.border}`;
      paint($("delta"), NEUTRAL);
    } else {
      const theme = s.b_is_better ? GOOD : BAD;
      const sign = s.delta_bpb < 0 ? "−" : "+";
      $("delta").textContent = `${sign}${Math.abs(s.delta_bpb).toFixed(4)}`;
      $("delta-pct").textContent = `${sign}${Math.abs(s.delta_pct).toFixed(2)}%`;
      $("at-iter").textContent = `· at iter ${s.comparison.iter.toLocaleString()}`;
      $("verdict-text").innerHTML = s.b_is_better
        ? `<b>${B.label}</b> has a lower observed BPB — ${Math.abs(s.delta_pct).toFixed(2)}% lower bits/byte than <b>${A.label}</b>`
        : `<b>${B.label}</b> has a higher observed BPB — ${Math.abs(s.delta_pct).toFixed(2)}% higher bits/byte than <b>${A.label}</b>`;
      $("verdict-text").innerHTML += "; single shared evaluation, no confidence interval. Cards show latest evaluations.";
      box.className = `mb-10 rounded-2xl border p-6 ${theme.border} ${theme.bg}`;
      paint($("delta"), theme);
      paint($("delta-pct"), theme);
      paint($("verdict-text"), theme);
    }

    const [pa, pb, lo, hi] = curves(A.history, B.history);
    $("line-a").setAttribute("d", pa);
    $("line-b").setAttribute("d", pb);
    if (lo !== null) { $("lo").textContent = `min ${f(lo, 3)}`; $("hi").textContent = `max ${f(hi, 3)}`; }
    $("npts").textContent = `· ${A.history.length} / ${B.history.length} points`;

    $("rows").innerHTML = s.comparison.pairs.slice().reverse().map(p => {
      const d = (p.b_bpb !== null && p.a_bpb !== null) ? p.b_bpb - p.a_bpb : null;
      const cls = d === null ? "text-zinc-400" : (d < 0 ? GOOD.text : BAD.text);
      const txt = d === null ? "–" : `${d < 0 ? "−" : "+"}${Math.abs(d).toFixed(4)}`;
      return `<tr class="border-b border-zinc-100 dark:border-white/5">
        <td class="py-1.5">${p.iter.toLocaleString()}</td>
        <td class="py-1.5 text-right">${f(p.a_bpb)}</td>
        <td class="py-1.5 text-right">${f(p.b_bpb)}</td>
        <td class="py-1.5 text-right font-medium ${cls}">${txt}</td></tr>`;
    }).join("");

    const running = !A.finished || !B.finished;
    $("live").innerHTML = running
      ? `<span class="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500"></span> ${s.now_ist} IST`
      : '<span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span> both complete';
    $("note").textContent = "Lower bits/byte is better. Both runs share data, schedule, "
      + "learning rates and seed; the only difference is the setting under test.";
  } catch (e) {
    $("live").textContent = "disconnected";
  }
}
tick();
setInterval(tick, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ohara-ab"
    protocol_version = "HTTP/1.1"
    path_a: Path
    path_b: Path
    label_a: str
    label_b: str
    total: int | None

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
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/status":
            payload = build_payload(
                self.path_a, self.path_b, self.label_a, self.label_b, self.total
            )
            self._send(json.dumps(payload).encode("utf-8"), "application/json")
        else:
            self._send(b'{"error":"not found"}', "application/json", status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an A/B training comparison")
    parser.add_argument("--a", required=True, help="baseline log")
    parser.add_argument("--b", required=True, help="log for the variant under test")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--total-iters", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        type(
            "Bound",
            (Handler,),
            {
                "path_a": Path(args.a),
                "path_b": Path(args.b),
                "label_a": args.label_a,
                "label_b": args.label_b,
                "total": args.total_iters,
            },
        ),
    )
    print(f"A/B dashboard on http://{args.host}:{args.port}")
    print(f"  A = {args.label_a}: {args.a}")
    print(f"  B = {args.label_b}: {args.b}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
