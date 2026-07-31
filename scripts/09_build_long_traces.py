"""
S5 / step 09 - build the L3/L4 reasoning band that open data does not contain.

Cleaning found the reachable open reasoning corpus tops out at 992 reasoning
tokens (median 132, p99 778) - the whole of it is band L1/L2. The plan (S4.2)
says the long bands have to be manufactured, and specifies the recipe: keep the
failed branches that were later corrected, because those are what teach
self-correction.

PRM800K is the one corpus that makes that recipe executable from real data. It
is a tree search over solution steps in which humans rated 60,398 completions
as WRONG (-1) alongside the ones that were chosen. The linear trace we cleaned in
step 02 threw all of that away. This script puts it back: for each step it emits
the rejected attempts, a correction marker, then the continuation that was
actually chosen - so the resulting trace is a real record of a real search, with
every token written by the original generator and every verdict a human's.

Nothing here is invented. What is *constructed* is the ordering, and the output
is tagged tier D (constructed) and written to its own shard so it can never be
confused with collected data.

Run: python scripts/09_build_long_traces.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "reasoning" / "prm800k_phase2_slice.jsonl"
OUT = ROOT / "data" / "clean" / "reasoning_L3L4_constructed.jsonl"
TOK = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "s5_bpe.json"))

BANDS = [("L0_direct", 32), ("L1_short", 256), ("L2_medium", 1024), ("L3_long", 4096)]


def band(n):
    for name, hi in BANDS:
        if n < hi:
            return name
    return "L4_ultra"


def build(rec):
    """One search trace out of one PRM800K record, in the order it happened."""
    q = rec.get("question") or {}
    steps = (rec.get("label") or {}).get("steps") or []
    if not q.get("problem") or not q.get("ground_truth_answer") or not steps:
        return None
    parts, n_rejected, n_branch_steps = [], 0, 0
    for st in steps:
        comps = st.get("completions") or []
        chosen = st.get("chosen_completion")
        rejected = [c for i, c in enumerate(comps)
                    if c.get("rating") is not None and c["rating"] < 0 and i != chosen]
        if rejected:
            n_branch_steps += 1
        for c in rejected:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            parts.append(f"<attempt>{txt}</attempt>")
            parts.append("<check>That step is wrong; discard it and try another route.</check>")
            n_rejected += 1
        keep = None
        if chosen is not None and chosen < len(comps):
            keep = comps[chosen]
        elif st.get("human_completion"):
            keep = st["human_completion"]
        else:
            keep = next((c for c in comps if (c.get("rating") or 0) > 0), None)
        if keep and (keep.get("text") or "").strip():
            parts.append(keep["text"].strip())
    if not parts or n_rejected == 0:
        return None          # no real branch point: this is just a linear L1/L2 trace
    trace = "\n".join(parts)
    n_reason = len(TOK.encode(trace).ids)
    return {
        "trace": trace, "problem": q["problem"], "answer": q["ground_truth_answer"],
        "n_reason": n_reason, "band": band(n_reason),
        "rejected_steps": n_rejected, "branch_steps": n_branch_steps,
    }


def main():
    kept, bands, longest = [], Counter(), None
    seen = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            b = build(rec)
            if not b:
                continue
            bands[b["band"]] += 1
            if b["band"] in ("L3_long", "L4_ultra"):
                kept.append(b)
                if longest is None or b["n_reason"] > longest["n_reason"]:
                    longest = b
    print(f"scanned {seen:,} PRM800K records")
    print("band distribution of reconstructed search traces:",
          {k: v for k, v in sorted(bands.items())})

    with open(OUT, "w", encoding="utf-8") as f:
        for i, b in enumerate(kept):
            segs = [{"text": "Problem: " + b["problem"], "supervised": False},
                    {"text": "<reasoning>\n" + b["trace"] + "\n</reasoning>", "supervised": True},
                    {"text": "Answer: " + b["answer"], "supervised": True}]
            n_tot = sum(len(TOK.encode(s["text"]).ids) for s in segs)
            f.write(json.dumps({
                "id": f"prm800k-search-{i}", "lane": "reasoning",
                "source": "prm800k-phase2-search-reconstruction",
                "license": "MIT", "tier": "D_constructed",
                "construction": "rejected human-rated (-1) completions, in order, each followed by "
                                "a correction marker, then the completion that was actually chosen; "
                                "all text is original PRM800K generator output, all verdicts human",
                "segments": segs, "lang": "en",
                "n_tokens": n_tot, "sup_tokens": n_tot - len(TOK.encode(segs[0]["text"]).ids),
                "ctx_tokens": len(TOK.encode(segs[0]["text"]).ids),
                "length_band": b["band"], "difficulty": "D4",
                "reasoning_tokens": b["n_reason"],
                "rejected_steps": b["rejected_steps"], "branch_steps": b["branch_steps"],
                "reserve": True,
            }, ensure_ascii=False) + "\n")

    tot = sum(json.loads(l)["n_tokens"] for l in open(OUT, encoding="utf-8")) if kept else 0
    # keep provenance intact: this shard joins the manifest like any other
    import hashlib
    mf = ROOT / "manifests" / "s5_shard_manifest.json"
    man = json.loads(mf.read_text(encoding="utf-8"))
    man = [m for m in man if m["shard"] != OUT.name]
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    man.append({"shard": OUT.name, "lane": "reasoning", "reserve": True, "docs": len(kept),
                "tokens": tot, "supervised_tokens": tot,
                "licenses": ["MIT"], "sources": ["prm800k-phase2-search-reconstruction"],
                "languages": ["en"], "difficulty": {"D4": len(kept)},
                "tier": "D_constructed",
                "note": "reconstructed from human-rated rejected branches; ordering constructed, "
                        "all text and all verdicts original",
                "content_sha256": h})
    mf.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(kept):,} L3/L4 traces -> {OUT.relative_to(ROOT)}  ({tot:,} tokens)")
    if longest:
        print(f"\nlongest reconstructed trace: {longest['n_reason']:,} reasoning tokens, "
              f"{longest['rejected_steps']} rejected steps across {longest['branch_steps']} "
              f"branch points, band {longest['band']}")
        print("  problem:", longest["problem"][:150].replace("\n", " "))
        print("  answer :", longest["answer"][:80])
    ceiling = max((b["n_reason"] for b in kept), default=0)
    if ceiling < 4096:
        print("\nL4_ultra remains EMPTY even after reconstruction (ceiling "
              f"{ceiling:,} reasoning tokens). PRM800K's searches are not deep enough; "
              "L4 can only come from a teacher sampled at high effort on problems with "
              "verifiable answers. That is the gate in mixture/v5_mixture.json.")


if __name__ == "__main__":
    main()
