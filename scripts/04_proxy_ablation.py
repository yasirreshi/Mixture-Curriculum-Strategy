"""
S5 / step 04 - the proxy run, actually run.

The plan's stage-1 protocol is 8 arms x 1B params x 30B tokens. That is 1.4e21
FLOPs and needs a GPU cluster. This machine has no GPU, so what runs here is a
reduced version of the same instrument: several small transformers, each trained
on the SAME token budget but a DIFFERENT mixture of the seven cleaned lanes, then
evaluated on a per-lane held-out set.

The method is RegMix's (arXiv:2407.01492), which fits its regression on 512 models
of 1M params x 1B tokens and then extrapolates 1000x. Ours is smaller than that,
so be clear about what it can and cannot do:

  CAN   - show that mixture changes per-lane loss in a measurable, ordered way
        - measure which lanes trade off against which (the interference matrix)
        - refute "the mixture does not matter"
        - rank arms on a capability-weighted objective
  CANNOT- confirm the absolute shares for a 40B run. Nothing at this scale can.
          That is what the 1B/3B protocol in mixture/v5_mixture.json is for.

Run: python scripts/04_proxy_ablation.py [--tokens 8000000] [--arms all]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results" / "proxy"
RESULTS.mkdir(parents=True, exist_ok=True)
TOKENIZER = ROOT / "data" / "tokenizer" / "s5_bpe.json"

LANES = ["code", "general_web", "indic", "stem_math", "reasoning", "long_context", "agentic"]

# ---------------------------------------------------------------- arms
# Every arm spends the same token budget. Only the composition differs.
ARMS = {
    "v5_proposed":   {"code": 24.0, "general_web": 31.0, "indic": 17.9, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "s5_composer_default": {"code": 24.0, "general_web": 34.0, "indic": 16.0, "stem_math": 12.0,
                            "reasoning": 6.0, "long_context": 6.0, "agentic": 2.0},
    "naive_web_heavy": {"code": 8.0, "general_web": 78.0, "indic": 6.0, "stem_math": 4.0,
                        "reasoning": 2.0, "long_context": 2.0, "agentic": 0.0},
    "code_heavy":    {"code": 32.0, "general_web": 23.0, "indic": 17.9, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "indic_lite":    {"code": 24.0, "general_web": 41.0, "indic": 8.0, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "indic_heavy":   {"code": 20.0, "general_web": 23.0, "indic": 30.0, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    # isolates the single disagreement between this plan and the session composer:
    # V5 shares with agentic raised from 1.0% to 2.0%, taken out of general web.
    "v5_agentic_2pct": {"code": 24.0, "general_web": 30.0, "indic": 17.9, "stem_math": 11.5,
                        "reasoning": 7.0, "long_context": 7.6, "agentic": 2.0},
}

# capability weights for the headline objective W (from mixture/v5_mixture.json)
W = {"code": 0.30, "agentic": 0.20, "indic": 0.20, "reasoning": 0.15,
     "long_context": 0.10, "general_web": 0.05, "stem_math": 0.0}


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=256, n_layer=4, n_head=4, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied, as in the V5 architecture
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        m = self.mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, m)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


# ---------------------------------------------------------------- data
def supply_caps(budget_tokens):
    """Per-lane unique-token caps, scaled from the real inventory to the proxy budget.

    Without this the ablation is structurally biased: every lane's stream is sampled
    with replacement, so giving a scarce lane a bigger share always lowers that lane's
    loss and never pays the repetition penalty that is the whole reason the plan holds
    the agentic share down. Scaling real unique supply by (proxy budget / 2.9T) makes
    an arm that overspends a scarce lane actually repeat it, and reproduces the real
    epoch counts almost exactly (agentic 10.6B -> ~11k tokens here).
    """
    inv = json.loads((ROOT / "mixture" / "inventory.json").read_text(encoding="utf-8"))
    scale = budget_tokens / 2.9e12
    caps = {}
    for lane in LANES:
        unique = inv["lanes"][lane]["lane_total_tokens"]
        if lane == "agentic":
            unique += inv["synthesis_capacity"]["agentic_verified_trajectories"]["tokens_per_run_window"]
        caps[lane] = max(2048, int(unique * scale))
    return caps


def build_lane_streams(tok, per_lane_tokens, ctx, seed=0, caps=None):
    """Tokenise each lane once into a flat array; arms then sample from these."""
    rng = random.Random(seed)
    streams, held = {}, {}
    for lane in LANES:
        buf, hbuf = [], []
        files = sorted(CLEAN.glob(f"{lane}*.jsonl"))
        docs = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    docs.append(line)
        rng.shuffle(docs)
        need_train = per_lane_tokens
        need_held = 120_000
        for line in docs:
            d = json.loads(line)
            text = "\n".join(s["text"] for s in d["segments"])
            ids = tok.encode(text).ids + [1]
            if len(hbuf) < need_held:
                hbuf.extend(ids)
            elif len(buf) < need_train:
                buf.extend(ids)
            else:
                break
        keep = need_train if caps is None else min(need_train, caps[lane])
        streams[lane] = np.array(buf[:keep], dtype=np.int32)
        held[lane] = np.array(hbuf[:need_held], dtype=np.int32)
        note = "" if caps is None else f"  (supply cap {caps[lane]:,})"
        print(f"  {lane:14s} train {len(streams[lane]):>10,}  held-out {len(held[lane]):>8,}{note}")
    return streams, held


def sample_batch(streams, weights, bs, ctx, rng):
    lanes = [l for l in LANES if weights.get(l, 0) > 0 and len(streams[l]) > ctx + 1]
    p = np.array([weights[l] for l in lanes], dtype=np.float64)
    p /= p.sum()
    picks = rng.choice(len(lanes), size=bs, p=p)
    xs, ys = [], []
    for i in picks:
        s = streams[lanes[i]]
        j = rng.integers(0, len(s) - ctx - 1)
        xs.append(s[j:j + ctx])
        ys.append(s[j + 1:j + ctx + 1])
    return (torch.from_numpy(np.stack(xs)).long(),
            torch.from_numpy(np.stack(ys)).long())


@torch.no_grad()
def eval_lanes(model, held, ctx, n_batches=12, bs=8, seed=1234):
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for lane, s in held.items():
        if len(s) < ctx + 2:
            continue
        tot, cnt = 0.0, 0
        for _ in range(n_batches):
            xs, ys = [], []
            for _ in range(bs):
                j = rng.integers(0, len(s) - ctx - 1)
                xs.append(s[j:j + ctx])
                ys.append(s[j + 1:j + ctx + 1])
            x = torch.from_numpy(np.stack(xs)).long()
            y = torch.from_numpy(np.stack(ys)).long()
            _, loss = model(x, y)
            tot += loss.item()
            cnt += 1
        out[lane] = tot / cnt
    model.train()
    return out


def train_arm(name, weights, streams, held, cfg):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = TinyLM(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"])
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    steps = cfg["tokens"] // (cfg["bs"] * cfg["ctx"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    t0 = time.time()
    curve = []
    for step in range(steps):
        x, y = sample_batch(streams, weights, cfg["bs"], cfg["ctx"], rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4)})
            el = time.time() - t0
            print(f"\r  {name:22s} step {step:>5}/{steps}  loss {loss.item():.3f}  "
                  f"{el:.0f}s  eta {el/(step+1)*(steps-step):.0f}s", end="")
    print()
    ev = eval_lanes(model, held, cfg["ctx"])
    return {"arm": name, "weights": weights, "params": n_par, "steps": steps,
            "tokens": steps * cfg["bs"] * cfg["ctx"], "wall_s": round(time.time() - t0, 1),
            "final_train_loss": curve[-1]["loss"] if curve else None,
            "heldout_nll": {k: round(v, 4) for k, v in ev.items()}, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6_000_000, help="training tokens per arm")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--supply-scaled", action="store_true",
                    help="cap each lane's unique stream at its real supply scaled to this budget, "
                         "so overspending a scarce lane pays a repetition penalty")
    ap.add_argument("--out", default="proxy_report.md")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild results/proxy_report.md from an existing runs.json, no training")
    a = ap.parse_args()
    if a.report_only:
        runs = json.loads((RESULTS / ("runs_supply_scaled.json" if a.supply_scaled
                                      else "runs.json")).read_text(encoding="utf-8"))
        return score_and_report(runs, a.ctx, a.out,
                                supply_caps(a.tokens) if a.supply_scaled else None)
    if a.threads:
        torch.set_num_threads(a.threads)

    tok = Tokenizer.from_file(str(TOKENIZER))
    cfg = {"vocab": tok.get_vocab_size(), "d": a.d, "layers": a.layers, "heads": 4,
           "ctx": a.ctx, "bs": a.bs, "lr": 3e-3, "tokens": a.tokens, "seed": 1337}

    print("building lane streams ...")
    # every arm must be able to draw its whole budget from any lane it weights
    per_lane = int(a.tokens * 0.85)
    caps = supply_caps(a.tokens) if a.supply_scaled else None
    streams, held = build_lane_streams(tok, per_lane, a.ctx, caps=caps)

    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    runs = []
    for name in arms:
        w = dict(ARMS[name])
        avail = {l: len(streams[l]) for l in LANES}
        # a lane with no data cannot be sampled; renormalise and record the fact
        missing = [l for l in w if w[l] > 0 and avail[l] < a.ctx + 2]
        if missing:
            print(f"  ! {name}: lanes with no data, reweighted away: {missing}")
            for l in missing:
                w[l] = 0.0
        s = sum(w.values())
        w = {k: 100 * v / s for k, v in w.items()}
        runs.append(train_arm(name, w, streams, held, cfg))
        (RESULTS / ("runs_supply_scaled.json" if a.supply_scaled else "runs.json")).write_text(
            json.dumps(runs, indent=2), encoding="utf-8")

    return score_and_report(runs, a.ctx, a.out, caps)


def score_and_report(runs, ctx, out_name="proxy_report.md", caps=None):
    runs_name = "runs_supply_scaled.json" if caps else "runs.json"
    # ---------------- scoring
    best = {l: min(r["heldout_nll"].get(l, 9e9) for r in runs) for l in LANES}
    for r in runs:
        rel = {l: (r["heldout_nll"].get(l, float("nan")) / best[l]) for l in LANES if best[l] < 9e9}
        r["relative_nll"] = {k: round(v, 4) for k, v in rel.items()}
        r["W"] = round(sum(W[l] * rel[l] for l in rel if W.get(l)), 5)
    runs.sort(key=lambda r: r["W"])
    (RESULTS / runs_name).write_text(json.dumps(runs, indent=2), encoding="utf-8")

    # ---------------- report
    L = ["# Proxy ablation - what actually ran\n"]
    L.append(f"_{len(runs)} arms x {runs[0]['params']:,} params x {runs[0]['tokens']:,} tokens, "
             f"context {ctx}, identical seed and identical token budget; only the mixture differs. "
             f"CPU, {sum(r['wall_s'] for r in runs)/60:.1f} minutes total._\n")
    L.append("**W** = capability-weighted held-out NLL, each lane normalised to the best arm "
             "(lower is better): " + " + ".join(f"{v:.2f}*{k}" for k, v in W.items() if v) + ".\n")
    L.append("| arm | W | " + " | ".join(LANES) + " |")
    L.append("|---|---:|" + "---:|" * len(LANES))
    for r in runs:
        L.append(f"| `{r['arm']}` | **{r['W']:.4f}** | "
                 + " | ".join(f"{r['heldout_nll'].get(l, float('nan')):.3f}" for l in LANES) + " |")
    L.append("\n(cells are held-out cross-entropy in nats/token on that lane's decontaminated "
             "hold-out, which no arm trained on)\n")
    L.append("> Do not compare cells *across* lanes. A lane's absolute loss is confounded by its "
             "tokenizer fertility and by how templated its sources are - Indic Wikipedia is "
             "cheaper per token than English Wikipedia here largely because our 32k BPE spends "
             "3.6-7.5 tokens on an Indic word and 1.8 on an English one. `W` normalises each lane "
             "to the best arm on that same lane, so the confound cancels; the raw column does not.\n")

    L.append("## Spread: does the mixture matter at all?\n")
    L.append("| lane | best arm | worst arm | spread (nats) | spread (%) |")
    L.append("|---|---|---|---:|---:|")
    for l in LANES:
        vals = [(r["heldout_nll"].get(l), r["arm"]) for r in runs if l in r["heldout_nll"]]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        L.append(f"| {l} | {lo[1]} ({lo[0]:.3f}) | {hi[1]} ({hi[0]:.3f}) | "
                 f"{hi[0]-lo[0]:.3f} | {100*(hi[0]-lo[0])/lo[0]:.1f}% |")
    L.append("")
    (ROOT / "results" / out_name).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-20:]))
    print(f"\n-> {ROOT / 'results' / out_name}  ({RESULTS / runs_name})")




if __name__ == "__main__":
    main()
