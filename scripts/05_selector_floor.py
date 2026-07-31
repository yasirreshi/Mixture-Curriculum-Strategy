"""
S5 / step 05 - measuring the selector bias that the always-on floor exists to fix.

Session 5 asserts two things about OPUS-style online data selection:

  (a) the selector defines "useful" through a proxy direction built from
      benchmarks that are overwhelmingly English and code, so it under-values
      native Indic text and unfamiliar agentic trajectories;
  (b) it scores only the first ~500 tokens of a candidate, which destroys long
      agentic trajectories and long reasoning traces whose payload is at the end.

Both are claims about numbers, so this measures them instead of repeating them.

Method (a scaled-down but mechanically faithful OPUS):
  1. train a small base model on a balanced mixture (the "current model state");
  2. build a proxy direction g_proxy = mean gradient over benchmark-like batches
     (English web + code + STEM), exactly the bias the session describes;
  3. for every candidate batch from every lane, compute g_cand and score it by
     first-order utility g_cand . g_proxy - how much a step on this batch would
     reduce the proxy loss;
  4. keep the top 40% (V4's retained fraction) and read off per-lane retention;
  5. repeat with candidates truncated to their first PREFIX tokens vs a window
     drawn from anywhere in the document, to isolate effect (b);
  6. convert retention into realized lane share, and solve for the floor level
     that keeps realized share within 1 point of the scheduled share.

Run: python scripts/05_selector_floor.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("proxy", ROOT / "scripts" / "04_proxy_ablation.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

RESULTS = ROOT / "results" / "selector"
RESULTS.mkdir(parents=True, exist_ok=True)
LANES = P.LANES

CTX = 512
KEEP_FRACTION = 0.40          # V4 retained ~40% of candidate data
PREFIX = 500                  # the session's "we only check the initial 500 tokens"
N_CANDIDATES_PER_LANE = 90
SCHEDULED = {"code": 24.0, "general_web": 31.0, "indic": 17.9, "stem_math": 11.5,
             "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0}
# what the proxy direction is built from: English-and-code weighted, as described
PROXY_MIX = {"general_web": 0.45, "code": 0.40, "stem_math": 0.15}


def flat_grad(model, x, y):
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    return g, loss.item()


def make_batch(stream, ctx, rng, bs=2, prefix_only=False, doc_starts=None):
    xs, ys = [], []
    for _ in range(bs):
        if prefix_only and doc_starts is not None and len(doc_starts) > 1:
            j = int(rng.choice(doc_starts[:-1]))          # the head of a document
            j = min(j, len(stream) - ctx - 2)
        else:
            j = int(rng.integers(0, max(1, len(stream) - ctx - 1)))
        xs.append(stream[j:j + ctx])
        ys.append(stream[j + 1:j + ctx + 1])
    return (torch.from_numpy(np.stack(xs)).long(), torch.from_numpy(np.stack(ys)).long())


def build_streams_with_doc_starts(tok, per_lane_tokens):
    """Same lane streams as the ablation, but we also remember where each document
    begins, so 'the first 500 tokens' means the head of a real document."""
    streams, starts = {}, {}
    rng = np.random.default_rng(7)
    for lane in LANES:
        buf, st = [], []
        docs = []
        for fp in sorted((ROOT / "data" / "clean").glob(f"{lane}*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                docs += f.readlines()
        rng.shuffle(docs)
        for line in docs:
            d = json.loads(line)
            text = "\n".join(s["text"] for s in d["segments"])
            ids = tok.encode(text).ids
            if len(ids) < 64:
                continue
            st.append(len(buf))
            buf.extend(ids + [1])
            if len(buf) >= per_lane_tokens:
                break
        streams[lane] = np.array(buf, dtype=np.int32)
        starts[lane] = np.array(st, dtype=np.int64)
        print(f"  {lane:14s} {len(buf):>9,} tokens  {len(st):>6,} docs")
    return streams, starts


def main():
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "s5_bpe.json"))
    print("building streams ...")
    streams, starts = build_streams_with_doc_starts(tok, 2_500_000)

    cfg = {"vocab": tok.get_vocab_size(), "d": 256, "layers": 4, "heads": 4, "ctx": CTX,
           "bs": 8, "lr": 3e-3, "tokens": 2_000_000, "seed": 4242}
    balanced = {l: 100.0 / len(LANES) for l in LANES}
    print("\ntraining the base model (the 'current model state' the selector scores against) ...")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = P.TinyLM(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], CTX)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    steps = cfg["tokens"] // (cfg["bs"] * CTX)
    t0 = time.time()
    for step in range(steps):
        x, y = P.sample_batch(streams, balanced, cfg["bs"], CTX, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"\r  step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", end="")
    print()

    # ---------------- proxy direction
    print("building the proxy direction (English web + code + STEM) ...")
    g_proxy = None
    for lane, w in PROXY_MIX.items():
        for _ in range(8):
            x, y = make_batch(streams[lane], CTX, rng, bs=2)
            g, _ = flat_grad(model, x, y)
            g = g * w
            g_proxy = g.clone() if g_proxy is None else g_proxy + g
    g_proxy = g_proxy / g_proxy.norm()

    # ---------------- score candidates
    results = {}
    for mode, prefix_only in (("prefix_500", True), ("full_document_window", False)):
        scores = defaultdict(list)
        t0 = time.time()
        for lane in LANES:
            if len(streams[lane]) < CTX + 4:
                continue
            for i in range(N_CANDIDATES_PER_LANE):
                x, y = make_batch(streams[lane], min(CTX, PREFIX) if prefix_only else CTX,
                                  rng, bs=2, prefix_only=prefix_only,
                                  doc_starts=starts[lane])
                g, _ = flat_grad(model, x, y)
                util = float(torch.dot(g, g_proxy))
                cos = util / (float(g.norm()) + 1e-12)
                scores[lane].append({"utility": util, "cosine": cos})
            print(f"  {mode:22s} {lane:14s} done ({time.time()-t0:.0f}s)")
        # global top-40% retention across the pooled candidate set
        pooled = [(s["cosine"], lane) for lane, lst in scores.items() for s in lst]
        pooled.sort(reverse=True)
        keep_n = int(len(pooled) * KEEP_FRACTION)
        kept = pooled[:keep_n]
        kept_by_lane = defaultdict(int)
        for _, lane in kept:
            kept_by_lane[lane] += 1
        results[mode] = {
            "mean_cosine": {l: float(np.mean([s["cosine"] for s in scores[l]])) for l in scores},
            "retention_pct": {l: 100.0 * kept_by_lane[l] / len(scores[l]) for l in scores},
            "n_candidates_per_lane": N_CANDIDATES_PER_LANE,
        }

    # ---------------- realized shares and the floor that fixes them
    report = {"keep_fraction": KEEP_FRACTION, "prefix_tokens": PREFIX,
              "proxy_mix": PROXY_MIX, "scheduled_share_pct": SCHEDULED, "modes": results}
    for mode, r in results.items():
        ret = r["retention_pct"]
        raw = {l: SCHEDULED[l] * ret.get(l, 0.0) / 100.0 for l in SCHEDULED}
        tot = sum(raw.values()) or 1.0
        realized = {l: 100.0 * v / tot for l, v in raw.items()}
        r["realized_share_pct"] = realized
        r["share_delta_pt"] = {l: realized[l] - SCHEDULED[l] for l in SCHEDULED}
        # floor f_l needed so that the protected part alone keeps realized >= scheduled-1
        need = {}
        for l in ("indic", "agentic", "reasoning"):
            target = SCHEDULED[l] - 1.0
            rr = ret.get(l, 0.0) / 100.0
            # realized = f + (scheduled - f) * rr  (protected part passes untouched)
            f = (target - SCHEDULED[l] * rr) / max(1e-9, (1 - rr))
            need[l] = max(0.0, round(min(f, SCHEDULED[l]), 2))
        r["floor_required_pct"] = need

    (RESULTS / "selector_measurements.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # ---------------- markdown
    L = ["# Selector bias, measured\n"]
    L.append(f"_Base model {sum(p.numel() for p in model.parameters()):,} params trained on a "
             f"balanced 7-lane mixture; proxy direction built from "
             f"{', '.join(f'{k} {v:.0%}' for k, v in PROXY_MIX.items())}; "
             f"{N_CANDIDATES_PER_LANE} candidate batches per lane; top "
             f"{KEEP_FRACTION:.0%} retained (V4's retained fraction)._\n")
    for mode, r in results.items():
        L.append(f"## scoring on the **{mode.replace('_',' ')}**\n")
        L.append("| lane | mean cosine to proxy | retained @top-40% | scheduled share | "
                 "realized share | delta |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for l in LANES:
            if l not in r["mean_cosine"]:
                continue
            L.append(f"| {l} | {r['mean_cosine'][l]:+.4f} | {r['retention_pct'][l]:.1f}% | "
                     f"{SCHEDULED[l]:.1f}% | {r['realized_share_pct'][l]:.1f}% | "
                     f"{r['share_delta_pt'][l]:+.1f}pt |")
        L.append("")
        L.append("Floor needed to hold realized share within 1pt of scheduled: "
                 + ", ".join(f"**{k} {v:.2f}%**" for k, v in r["floor_required_pct"].items()) + "\n")
    (ROOT / "results" / "selector_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
