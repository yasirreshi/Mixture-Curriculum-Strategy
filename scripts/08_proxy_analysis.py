"""
S5 / step 08 - turn the proxy runs into a statement about the mixture.

RegMix's actual contribution is not "train small models" but "fit a regression from
mixture weights to loss, then use the fit to choose". With only a handful of arms a
full 7-dimensional fit is underdetermined, so this fits the one relationship each
arm actually varies: a lane's own share against that lane's held-out loss,

    nll_l  =  a_l + b_l * ln(share_l)

and reports b_l - the nats per e-fold of share, i.e. how much a lane pays for being
squeezed. That number is what says whether a share is near a flat part of the curve
(cheap to cut) or a steep one (expensive to cut), which is exactly the question a
reviewer asks about every line of the mixture table.

It then does the only thing that makes the fit useful: takes 3 points of share off
each lane, gives them to each other lane, and asks whether any single transfer
improves the capability-weighted objective W. If one does, the mixture is not at a
local optimum and the plan should say so.

Run: python scripts/08_proxy_analysis.py
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RUNS = json.loads((ROOT / "results" / "proxy" / "runs.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "proxy_report.md"

LANES = ["code", "general_web", "indic", "stem_math", "reasoning", "long_context", "agentic"]
W = {"code": 0.30, "agentic": 0.20, "indic": 0.20, "reasoning": 0.15,
     "long_context": 0.10, "general_web": 0.05, "stem_math": 0.0}


def fit(xs, ys):
    """least squares y = a + b x"""
    n = len(xs)
    if n < 3:
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main():
    fits = {}
    for lane in LANES:
        pts = [(r["weights"].get(lane, 0.0), r["heldout_nll"].get(lane))
               for r in RUNS if r["heldout_nll"].get(lane) is not None]
        pts = [(s, l) for s, l in pts if s > 0.05]
        if len(pts) < 3:
            continue
        xs = [math.log(s) for s, _ in pts]
        ys = [l for _, l in pts]
        a, b, r2 = fit(xs, ys)
        if b is None:
            continue
        fits[lane] = {"a": a, "b": b, "r2": r2, "n": len(pts),
                      "share_range": [min(s for s, _ in pts), max(s for s, _ in pts)]}

    v5 = next((r for r in RUNS if r["arm"] == "v5_proposed"), None)
    base = v5["weights"] if v5 else None

    # predicted W under a share transfer, using the fitted curves
    transfers = []
    if base:
        best = {l: min(r["heldout_nll"].get(l, 9e9) for r in RUNS) for l in LANES}

        def predW(weights):
            tot = 0.0
            for l, w in W.items():
                if not w or l not in fits or weights.get(l, 0) <= 0.05:
                    continue
                f = fits[l]
                nll = f["a"] + f["b"] * math.log(weights[l])
                tot += w * nll / best[l]
            return tot

        w0 = predW(base)
        for src in LANES:
            for dst in LANES:
                if src == dst or base.get(src, 0) < 4.0:
                    continue
                w = dict(base)
                w[src] -= 3.0
                w[dst] += 3.0
                transfers.append((predW(w) - w0, src, dst))
        transfers.sort()

    L = []
    prev = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if prev:
        L.append(prev.rstrip() + "\n")
    L.append("## Own-share elasticity: what each lane pays for being squeezed\n")
    L.append("`nll_lane = a + b·ln(share_lane)`, fitted across the arms that gave the lane a "
             "non-zero share. **b is nats of held-out loss per e-fold of share** — more negative "
             "means the lane is on a steep part of the curve and is expensive to cut.\n")
    L.append("| lane | b (nats per e-fold) | R² | arms | share range tested |")
    L.append("|---|---:|---:|---:|---|")
    for lane in sorted(fits, key=lambda l: fits[l]["b"]):
        f = fits[lane]
        L.append(f"| {lane} | **{f['b']:+.3f}** | {f['r2']:.2f} | {f['n']} | "
                 f"{f['share_range'][0]:.1f}% – {f['share_range'][1]:.1f}% |")
    L.append("")
    if "long_context" in fits and abs(fits["long_context"]["b"]) < 0.1:
        L.append("> The long-context row measures the instrument, not the lane. At context 256 a "
                 "model cannot express long-context capability at all, so its loss barely responds "
                 "to share. Read it as *this proxy is blind to this lane* - which is worth knowing "
                 "before spending 1B-scale compute on the same design: the stage-1 arms have to run "
                 "at a context long enough for the lane to mean anything.\n")

    if transfers:
        L.append("## Is the mixture at a local optimum?\n")
        L.append("Moving **3 points of share** from one lane to another and predicting the change "
                 "in the capability-weighted objective `W` from the fitted curves. Negative = the "
                 "transfer would improve the plan.\n")
        L.append("| transfer | ΔW |")
        L.append("|---|---:|")
        for d, s, t in transfers[:5]:
            L.append(f"| {s} → {t} | **{d:+.4f}** |")
        L.append("| … | |")
        for d, s, t in transfers[-3:]:
            L.append(f"| {s} → {t} | {d:+.4f} |")
        L.append("")
        d, s, t = transfers[0]
        if d < -1e-4:
            L.append(f"> **The proxy does not endorse the mixture as written.** Its best single "
                     f"move is 3 points from `{s}` to `{t}` (ΔW = {d:+.4f}). At this scale that is "
                     f"a direction, not a decision — the fit has {len(RUNS)} points, the models are "
                     f"11.4M parameters, and the agentic slope is fitted over a 1–2% range in which "
                     f"repetition is free (§8.3 removes that) — but it is exactly the hypothesis "
                     f"the 1B arm list in §8.1 has to settle, and it is written down here before "
                     f"the 1B runs rather than after.\n")
        else:
            L.append("> No single 3-point transfer improves `W` under the fitted curves: at this "
                     "scale the mixture sits at a local optimum of the objective it was designed "
                     "for.\n")

    L.append("## What this proxy is not\n")
    L.append("Three orders of magnitude below the pre-registered 1B protocol, on 11.4M-parameter "
             "models trained for 3.0M tokens each. It cannot rank arms on downstream benchmarks — "
             "nothing at this size produces a non-trivial HumanEval or MILU score — and RegMix's "
             "own caution applies doubly here: domain interactions are complicated and small-model "
             "orderings can invert. It is reported because it is real, it was cheap, it is "
             "reproducible from this repo, and it puts numbers where the plan would otherwise have "
             "adjectives.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-40:]))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
