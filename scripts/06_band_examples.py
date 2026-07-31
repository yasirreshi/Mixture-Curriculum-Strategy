"""
S5 / step 06 - a real example for every difficulty band and every reasoning-length band.

The assignment asks for a concrete example at each level. Inventing them would be
easy and worthless; these are pulled out of the corpus this repo actually cleaned,
with their measured token counts, so each band is anchored to data that exists.

Run: python scripts/06_band_examples.py
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "results" / "band_examples.md"

DIFF_ORDER = ["D0", "D1", "D2", "D3", "D4"]
LEN_ORDER = ["L0_direct", "L1_short", "L2_medium", "L3_long", "L4_ultra"]
DIFF_NAME = {"D0": "Nursery", "D1": "School", "D2": "High school / undergrad",
             "D3": "Graduate", "D4": "Frontier / PhD"}


def snippet(d, n=420):
    text = "\n".join(s["text"] for s in d["segments"])
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return (text[:n] + (" ..." if len(text) > n else "")).replace("|", "\\|")


def main():
    by_diff = defaultdict(list)
    by_len = defaultdict(list)
    rng = random.Random(11)
    for fp in sorted(CLEAN.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                by_diff[d["difficulty"]].append(d)
                if d.get("length_band"):
                    by_len[d["length_band"]].append(d)

    L = ["# Difficulty and reasoning-length bands, with real examples\n",
         "_Every example below is a document from `data/clean/` in this repo, with the "
         "token count measured by the 32k BPE trained in `scripts/02_clean_lanes.py`. "
         "Band assignment is done by the documented heuristics in that script, not by hand._\n"]

    L.append("## Difficulty ladder\n")
    L.append("| band | name | docs in corpus | lanes present | median tokens |")
    L.append("|---|---|---:|---|---:|")
    for b in DIFF_ORDER:
        ds = by_diff.get(b, [])
        if not ds:
            L.append(f"| **{b}** | {DIFF_NAME[b]} | 0 | - | - |")
            continue
        lanes = sorted({d["lane"] for d in ds})
        toks = sorted(d["n_tokens"] for d in ds)
        L.append(f"| **{b}** | {DIFF_NAME[b]} | {len(ds):,} | {', '.join(lanes)} | "
                 f"{toks[len(toks)//2]:,} |")
    L.append("")

    for b in DIFF_ORDER:
        ds = by_diff.get(b, [])
        if not ds:
            continue
        L.append(f"### {b} - {DIFF_NAME[b]}\n")
        # prefer variety: one example from up to two different lanes
        seen_lanes = set()
        picks = []
        for d in rng.sample(ds, min(120, len(ds))):
            if d["lane"] in seen_lanes:
                continue
            seen_lanes.add(d["lane"])
            picks.append(d)
            if len(picks) == 2:
                break
        for d in picks:
            L.append(f"- **lane** `{d['lane']}` · **source** `{d['source']}` · "
                     f"**{d['n_tokens']:,} tokens** "
                     f"({d['sup_tokens']:,} supervised / {d['ctx_tokens']:,} context)"
                     + (f" · lang `{d.get('lang')}`" if d.get("lang") else ""))
            L.append(f"  > {snippet(d)}\n")

    L.append("## Reasoning-length bands\n")
    L.append("| band | reasoning tokens | control token | docs in corpus | median total tokens |")
    L.append("|---|---|---|---:|---:|")
    ctrl = {"L0_direct": "<effort=none>", "L1_short": "<effort=low>", "L2_medium": "<effort=medium>",
            "L3_long": "<effort=high>", "L4_ultra": "<effort=ultra>"}
    rng_bounds = {"L0_direct": "0-32", "L1_short": "32-256", "L2_medium": "256-1024",
                  "L3_long": "1024-4096", "L4_ultra": "4096-32768"}
    for b in LEN_ORDER:
        ds = by_len.get(b, [])
        med = sorted(d["n_tokens"] for d in ds)[len(ds)//2] if ds else 0
        L.append(f"| **{b}** | {rng_bounds[b]} | `{ctrl[b]}` | {len(ds):,} | {med:,} |")
    L.append("")
    for b in LEN_ORDER:
        ds = by_len.get(b, [])
        if not ds:
            L.append(f"### {b}\n\n_No document in the cleaned corpus falls in this band. "
                     f"That is itself a finding: this band has to be manufactured._\n")
            continue
        d = max(rng.sample(ds, min(60, len(ds))), key=lambda z: z["n_tokens"])
        L.append(f"### {b} · `{ctrl[b]}`\n")
        L.append(f"- **source** `{d['source']}` · **{d['n_tokens']:,} tokens** "
                 f"({d['sup_tokens']:,} supervised) · difficulty `{d['difficulty']}`")
        L.append(f"  > {snippet(d, 700)}\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {OUT}")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
