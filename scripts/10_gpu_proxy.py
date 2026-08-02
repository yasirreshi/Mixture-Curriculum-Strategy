"""
S5 / step 10 - the proxy run with the corrections the plan pre-registered, on GPU.

Steps 04 and 04 --supply-scaled ran on CPU at 11.4M params / 3.0M tokens / context 256.
Both reports ended with the same admission: the instrument could not settle the agentic
share, for three specific reasons written into the stage-1 protocol as required corrections.

This script implements all three and runs the experiment.

  correction 1 (from 04)      per-lane unique-supply caps, so overspending a scarce
                              lane genuinely repeats it. Carried over, always on.
  correction 2 (from 08)      a context long enough for the long-context lane to be
                              measurable. 256 -> 1024.
  correction 3 (from 04-ss)   OUT-OF-DISTRIBUTION hold-outs for agentic and reasoning,
                              drawn from sources absent from training. This is the one
                              that matters: with an in-distribution hold-out, an arm that
                              over-repeats a small pool is rewarded for memorising that
                              pool's formatting, which is exactly the failure the 4-epoch
                              cap exists to prevent. The CPU runs could not tell the two
                              apart and said so; this one can.

Plus two things the CPU instrument did not do at all:

  loss masking                the plan's §4.1 convention - loss falls on the model's own
                              planning/tool-calls/answer, never on tool observations. The
                              cleaned corpus carries a per-segment `supervised` flag, so
                              --loss-mask trains the way the real run would.
  paired bootstrap CIs        the pre-registered confirm/refute rule is stated in terms of
                              non-overlapping bootstrap CIs. Held-out sequences are built
                              once and shared by every arm, so arms can be compared paired,
                              which is what makes a CI at this budget meaningful.

OOD hold-out construction (whole source families, never seen in training):

  agentic     train Gorilla (APIBench HF/TF + OpenFunctions)  hold out ALL BFCL v4
  reasoning   train GSM8K (plain + socratic)                  hold out PRM800K phase-2
  code        train sqlite/numpy/sklearn/fastapi/... (11 repos)  hold out redis, ripgrep,
                                                              cobra, fd, express
  others      in-distribution document split, declared as such - no OOD claim is made
              for web / Indic / STEM / long-context.

Run: python scripts/10_gpu_proxy.py --tokens 120000000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
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

# ---------------------------------------------------------------- OOD hold-out policy
# A lane's hold-out is OOD when it is drawn from source families that contribute ZERO
# tokens to training. Anything else is declared in-distribution and labelled as such.
OOD_HOLDOUT_SOURCES = {
    "agentic": ["bfcl-v4-live-multiple", "bfcl-v4-multi-turn", "bfcl-v4-parallel"],
    "reasoning": ["prm800k-phase2"],
    "code": ["redis/redis", "BurntSushi/ripgrep", "spf13/cobra", "sharkdp/fd", "expressjs/express"],
}

ARMS = {
    "v5_proposed":   {"code": 24.0, "general_web": 31.0, "indic": 17.9, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "v5_agentic_2pct": {"code": 24.0, "general_web": 30.0, "indic": 17.9, "stem_math": 11.5,
                        "reasoning": 7.0, "long_context": 7.6, "agentic": 2.0},
    "s5_composer_default": {"code": 24.0, "general_web": 34.0, "indic": 16.0, "stem_math": 12.0,
                            "reasoning": 6.0, "long_context": 6.0, "agentic": 2.0},
    "code_heavy":    {"code": 32.0, "general_web": 23.0, "indic": 17.9, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "indic_heavy":   {"code": 20.0, "general_web": 23.0, "indic": 30.0, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "indic_lite":    {"code": 24.0, "general_web": 41.0, "indic": 8.0, "stem_math": 11.5,
                      "reasoning": 7.0, "long_context": 7.6, "agentic": 1.0},
    "naive_web_heavy": {"code": 8.0, "general_web": 78.0, "indic": 6.0, "stem_math": 4.0,
                        "reasoning": 2.0, "long_context": 2.0, "agentic": 0.0},
}

# W as pre-registered in mixture/v5_mixture.json. stem_math carries weight 0 there; the
# report also prints W_stem, which gives it 0.05, because a metric blind to an 11.5% lane
# cannot adjudicate transfers out of that lane (see the local-optimum table in step 08).
W = {"code": 0.30, "agentic": 0.20, "indic": 0.20, "reasoning": 0.15,
     "long_context": 0.10, "general_web": 0.05, "stem_math": 0.0}
W_STEM = {"code": 0.285, "agentic": 0.19, "indic": 0.19, "reasoning": 0.1425,
          "long_context": 0.095, "general_web": 0.0475, "stem_math": 0.05}


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(b, t, self.h, d // self.h).transpose(1, 2)
        k = k.view(b, t, self.h, d // self.h).transpose(1, 2)
        v = v.view(b, t, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.mlp(self.ln2(x))


class ProxyLM(nn.Module):
    def __init__(self, vocab, d=512, n_layer=8, n_head=8, ctx=1024):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied, as in the V5 architecture
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


# ---------------------------------------------------------------- data
def supply_caps(budget_tokens):
    """Per-lane unique-token caps: real inventory supply scaled to the proxy budget."""
    inv = json.loads((ROOT / "mixture" / "inventory.json").read_text(encoding="utf-8"))
    scale = budget_tokens / 2.9e12
    caps = {}
    for lane in LANES:
        unique = inv["lanes"][lane]["lane_total_tokens"]
        if lane == "agentic":
            unique += inv["synthesis_capacity"]["agentic_verified_trajectories"]["tokens_per_run_window"]
        caps[lane] = max(4096, int(unique * scale))
    return caps


def encode_doc(tok, d, mask_loss):
    """Return (ids, loss_flags). loss_flags is 1 where the token carries loss.

    The plan's masking convention: the model's own planning / tool calls / answer carry
    loss; tool observations, repo files and user turns are context and carry none.
    """
    ids, flags = [], []
    for seg in d["segments"]:
        sid = tok.encode(seg["text"]).ids
        sup = seg.get("supervised")
        # a lane with no mask information trains on everything, as it does today
        keep = 1 if (not mask_loss or sup is None or sup) else 0
        ids.extend(sid)
        flags.extend([keep] * len(sid))
    ids.append(1)
    flags.append(1)
    return ids, flags


def build_streams(tok, budget, ctx, caps, mask_loss, seed=0, held_seqs=192):
    """Tokenise once. Training streams exclude every OOD hold-out source entirely."""
    rng = random.Random(seed)
    streams, flagstreams, held, held_kind = {}, {}, {}, {}
    for lane in LANES:
        ood_srcs = set(OOD_HOLDOUT_SOURCES.get(lane, []))
        train_docs, hold_docs = [], []
        for fp in sorted(CLEAN.glob(f"{lane}*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    (hold_docs if d.get("source") in ood_srcs else train_docs).append(d)
        if ood_srcs:
            held_kind[lane] = "OOD (" + ", ".join(sorted(ood_srcs)) + ")"
        else:
            # no OOD family available: fall back to a held-out document split, declared
            rng.shuffle(train_docs)
            n_hold = max(1, int(0.06 * len(train_docs)))
            hold_docs, train_docs = train_docs[:n_hold], train_docs[n_hold:]
            held_kind[lane] = "in-distribution (held-out documents)"

        rng.shuffle(train_docs)
        cap = caps[lane]
        buf, fbuf = [], []
        for d in train_docs:
            ids, fl = encode_doc(tok, d, mask_loss)
            buf.extend(ids)
            fbuf.extend(fl)
            if len(buf) >= cap:
                break
        streams[lane] = np.array(buf[:cap], dtype=np.int32)
        flagstreams[lane] = np.array(fbuf[:cap], dtype=np.int8)

        hbuf, hfbuf = [], []
        for d in hold_docs:
            ids, fl = encode_doc(tok, d, mask_loss)
            hbuf.extend(ids)
            hfbuf.extend(fl)
            if len(hbuf) >= held_seqs * (ctx + 1):
                break
        held[lane] = (np.array(hbuf, dtype=np.int32), np.array(hfbuf, dtype=np.int8))
        print(f"  {lane:14s} train {len(streams[lane]):>10,} (cap {cap:>10,})  "
              f"held-out {len(hbuf):>9,}  {held_kind[lane]}")
    return streams, flagstreams, held, held_kind


def make_held_batches(held, ctx, n_seq=160, seed=1234):
    """Fixed held-out sequences, identical for every arm -> paired comparisons."""
    rng = np.random.default_rng(seed)
    out = {}
    for lane, (s, f) in held.items():
        if len(s) < ctx + 2:
            continue
        n = min(n_seq, max(1, (len(s) - 1) // ctx))
        starts = rng.choice(max(1, len(s) - ctx - 1), size=n, replace=len(s) - ctx - 1 < n)
        x = np.stack([s[j:j + ctx] for j in starts])
        y = np.stack([s[j + 1:j + ctx + 1] for j in starts])
        m = np.stack([f[j + 1:j + ctx + 1] for j in starts])
        out[lane] = (torch.from_numpy(x).long(), torch.from_numpy(y).long(),
                     torch.from_numpy(m).float())
    return out


def to_gpu_streams(streams, flags, dev):
    """Keep the lane streams resident on the GPU; batch assembly is then a gather, not a
    per-step Python loop over numpy slices (which dominated wall time at context 1024)."""
    g = {}
    for l in streams:
        if len(streams[l]) < 8:
            continue
        g[l] = (torch.from_numpy(streams[l].astype(np.int64)).to(dev),
                torch.from_numpy(flags[l].astype(np.float32)).to(dev))
    return g


def sample_batch(gstreams, weights, bs, ctx, gen, dev):
    lanes = [l for l in LANES if weights.get(l, 0) > 0 and l in gstreams
             and len(gstreams[l][0]) > ctx + 1]
    p = torch.tensor([weights[l] for l in lanes], dtype=torch.float, device=dev)
    counts = torch.multinomial(p / p.sum(), bs, replacement=True, generator=gen)
    xs, ys, ms = [], [], []
    off = torch.arange(ctx + 1, device=dev)
    for i in range(len(lanes)):
        n = int((counts == i).sum())
        if n == 0:
            continue
        s, f = gstreams[lanes[i]]
        j = torch.randint(0, len(s) - ctx - 1, (n,), device=dev, generator=gen)
        idx = j[:, None] + off[None, :]              # (n, ctx+1)
        seq, fl = s[idx], f[idx]
        xs.append(seq[:, :-1])
        ys.append(seq[:, 1:])
        ms.append(fl[:, 1:])
    return torch.cat(xs), torch.cat(ys), torch.cat(ms)


def masked_loss(logits, y, m, use_mask):
    # no fp32 upcast of the full (bs*ctx, 32000) logit tensor: at bs=24/ctx=1024 that is a
    # 3.1GB temporary and it, not the matmuls, was the throughput bottleneck.
    if not use_mask:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                         reduction="none").view_as(y)
    return (ce * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def eval_lanes(model, batches, use_mask, dev, bs=8):
    """Per-SEQUENCE held-out NLL, kept as a vector so the CIs can be bootstrapped."""
    model.eval()
    out = {}
    for lane, (x, y, m) in batches.items():
        per_seq = []
        for i in range(0, len(x), bs):
            xb, yb, mb = x[i:i + bs].to(dev), y[i:i + bs].to(dev), m[i:i + bs].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xb)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), yb.reshape(-1),
                                 reduction="none").view_as(yb)
            if use_mask:
                num = (ce * mb).sum(1)
                den = mb.sum(1).clamp(min=1.0)
                per_seq.append((num / den).cpu())
            else:
                per_seq.append(ce.mean(1).cpu())
        out[lane] = torch.cat(per_seq).numpy().astype(np.float64)
    model.train()
    return out


def train_arm(name, weights, gstreams, batches, cfg, dev):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    model = ProxyLM(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["ctx"]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(dev.type == "cuda"))
    steps = cfg["tokens"] // (cfg["bs"] * cfg["ctx"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps,
                                                pct_start=0.05, anneal_strategy="cos")
    t0, curve = time.time(), []
    for step in range(steps):
        x, y, m = sample_batch(gstreams, weights, cfg["bs"], cfg["ctx"], gen, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x)
        loss = masked_loss(logits, y, m, cfg["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4)})
            el = time.time() - t0
            print(f"\r  {name:22s} step {step:>5}/{steps}  loss {loss.item():.3f}  "
                  f"{el:.0f}s  eta {el/(step+1)*(steps-step):.0f}s   ", end="", flush=True)
    print()
    ev = eval_lanes(model, batches, cfg["mask"], dev)
    del model, opt
    torch.cuda.empty_cache()
    return {"arm": name, "weights": weights, "params": n_par, "steps": steps,
            "tokens": steps * cfg["bs"] * cfg["ctx"], "wall_s": round(time.time() - t0, 1),
            "final_train_loss": curve[-1]["loss"] if curve else None,
            "heldout_nll": {k: float(v.mean()) for k, v in ev.items()},
            "heldout_per_seq": {k: [round(float(z), 5) for z in v] for k, v in ev.items()},
            "curve": curve}


# ---------------------------------------------------------------- scoring
def score(runs, weights):
    best = {l: min(r["heldout_nll"].get(l, 9e9) for r in runs) for l in LANES}
    for r in runs:
        rel = {l: r["heldout_nll"][l] / best[l] for l in LANES if l in r["heldout_nll"]}
        key = "W" if weights is W else "W_stem"
        r[key] = round(sum(weights[l] * rel[l] for l in rel if weights.get(l)), 5)
        if weights is W:
            r["relative_nll"] = {k: round(v, 4) for k, v in rel.items()}
    return runs


def paired_bootstrap(runs, weights, n_boot=4000, seed=7):
    """Resample held-out SEQUENCES (same indices for every arm) and re-score.

    Paired: arm A and arm B are always scored on the same resampled hold-out, so the CI on
    a *difference* reflects mixture effect rather than hold-out sampling noise. This is what
    the pre-registered rule ('non-overlapping bootstrap CIs') needs in order to mean anything.
    """
    rng = np.random.default_rng(seed)
    lanes = [l for l in LANES if l in runs[0]["heldout_per_seq"]]
    arr = {r["arm"]: {l: np.asarray(r["heldout_per_seq"][l]) for l in lanes} for r in runs}
    n = {l: len(arr[runs[0]["arm"]][l]) for l in lanes}
    Ws = {r["arm"]: [] for r in runs}
    for _ in range(n_boot):
        idx = {l: rng.integers(0, n[l], n[l]) for l in lanes}
        means = {a: {l: arr[a][l][idx[l]].mean() for l in lanes} for a in arr}
        best = {l: min(means[a][l] for a in arr) for l in lanes}
        for a in arr:
            Ws[a].append(sum(weights[l] * means[a][l] / best[l] for l in lanes if weights.get(l)))
    out = {}
    for a, v in Ws.items():
        v = np.asarray(v)
        out[a] = {"mean": float(v.mean()),
                  "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return out, {a: np.asarray(v) for a, v in Ws.items()}


def paired_lane_delta(runs, arm_a, arm_b, lane, n_boot=4000, seed=11):
    """CI on (arm_a - arm_b) held-out NLL for one lane, paired over sequences."""
    A = {r["arm"]: np.asarray(r["heldout_per_seq"][lane]) for r in runs}
    rng = np.random.default_rng(seed)
    n = len(A[arm_a])
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(A[arm_a][i].mean() - A[arm_b][i].mean())
    d = np.asarray(d)
    return {"delta": float(A[arm_a].mean() - A[arm_b].mean()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_a_better": float((d < 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=120_000_000)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--loss-mask", action="store_true",
                    help="apply the plan's per-segment loss mask (§4.1) during training and eval")
    ap.add_argument("--seed", type=int, default=1337,
                    help="training seed. Re-running one arm at several seeds measures the "
                         "instrument's own noise floor, which is what says whether a gap "
                         "between two arms means anything at all.")
    ap.add_argument("--out", default="proxy_report_gpu.md")
    ap.add_argument("--runs-name", default="runs_gpu.json")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        runs = json.loads((RESULTS / a.runs_name).read_text(encoding="utf-8"))
        return report(runs, a, json.loads((RESULTS / "gpu_meta.json").read_text(encoding="utf-8")))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"device: {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")

    tok = Tokenizer.from_file(str(TOKENIZER))
    cfg = {"vocab": tok.get_vocab_size(), "d": a.d, "layers": a.layers, "heads": a.heads,
           "ctx": a.ctx, "bs": a.bs, "lr": a.lr, "tokens": a.tokens, "seed": a.seed,
           "mask": a.loss_mask}
    caps = supply_caps(a.tokens)
    print(f"epoch-honest supply caps at a {a.tokens:,}-token budget:")
    for l in LANES:
        print(f"  {l:14s} {caps[l]:>12,}")
    print("building lane streams (OOD families excluded from training) ...")
    streams, flags, held, held_kind = build_streams(tok, a.tokens, a.ctx, caps, a.loss_mask)
    batches = make_held_batches(held, a.ctx)
    gstreams = to_gpu_streams(streams, flags, dev)

    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    runs = []
    for name in arms:
        w = dict(ARMS[name])
        missing = [l for l in w if w[l] > 0 and len(streams[l]) < a.ctx + 2]
        for l in missing:
            w[l] = 0.0
        if missing:
            print(f"  ! {name}: lanes with no data, reweighted away: {missing}")
        s = sum(w.values())
        w = {k: 100 * v / s for k, v in w.items()}
        # epochs this arm actually spends on each lane, given the caps
        ep = {l: round(a.tokens * w[l] / 100 / max(1, len(streams[l])), 2) for l in LANES}
        print(f"  epochs: " + "  ".join(f"{l[:4]}={ep[l]}" for l in LANES))
        r = train_arm(name, w, gstreams, batches, cfg, dev)
        r["epochs"] = ep
        r["seed"] = cfg["seed"]
        runs.append(r)
        (RESULTS / a.runs_name).write_text(json.dumps(runs, indent=2), encoding="utf-8")

    meta = {"held_kind": held_kind, "caps": caps, "cfg": {k: v for k, v in cfg.items()},
            "ood_sources": OOD_HOLDOUT_SOURCES,
            "held_seqs": {l: len(batches[l][0]) for l in batches}}
    (RESULTS / "gpu_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report(runs, a, meta)


def report(runs, a, meta):
    runs = score(runs, W)
    runs = score(runs, W_STEM)
    runs.sort(key=lambda r: r["W"])
    (RESULTS / a.runs_name).write_text(json.dumps(runs, indent=2), encoding="utf-8")

    ci, _ = paired_bootstrap(runs, W)
    names = [r["arm"] for r in runs]
    L = ["# Proxy ablation on GPU - the pre-registered corrections, run\n"]
    p = runs[0]
    L.append(f"_{len(runs)} arms x {p['params']:,} params x {p['tokens']:,} tokens, "
             f"context {meta['cfg']['ctx']}, identical seed and identical budget; only the "
             f"mixture differs. {sum(r['wall_s'] for r in runs)/60:.1f} GPU-minutes total "
             f"on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}._\n")
    L.append(f"Loss masking (§4.1 convention): **{'on' if meta['cfg']['mask'] else 'off'}**. "
             f"Per-lane unique-supply caps: **on** (epoch-honest).\n")

    L.append("## What each lane's hold-out actually is\n")
    L.append("| lane | hold-out | sequences |")
    L.append("|---|---|---:|")
    for l in LANES:
        if l in meta["held_kind"]:
            L.append(f"| {l} | {meta['held_kind'][l]} | {meta['held_seqs'].get(l, 0)} |")
    L.append("\n> The three lanes with an OOD hold-out are the three the plan said it could not "
             "adjudicate without one. For those, no training token comes from the source family "
             "the arm is scored on, so an arm cannot win by memorising a format. The other four "
             "are in-distribution document splits and are labelled as such - no OOD claim is "
             "made for them.\n")

    L.append("## Result\n")
    L.append("| rank | arm | W | 95% CI | W_stem | " + " | ".join(LANES) + " |")
    L.append("|---:|---|---:|---|---:|" + "---:|" * len(LANES))
    for i, r in enumerate(runs, 1):
        c = ci[r["arm"]]
        L.append(f"| {i} | `{r['arm']}` | **{r['W']:.4f}** | [{c['lo']:.4f}, {c['hi']:.4f}] | "
                 f"{r['W_stem']:.4f} | "
                 + " | ".join(f"{r['heldout_nll'].get(l, float('nan')):.3f}" for l in LANES) + " |")
    L.append("\n`W` is the pre-registered objective (stem_math weight 0). `W_stem` gives "
             "stem_math weight 0.05 with the others renormalised - reported because an "
             "objective blind to an 11.5% lane cannot price transfers out of it.\n")

    L.append("## Epochs actually spent (the caps biting)\n")
    L.append("| arm | " + " | ".join(LANES) + " |")
    L.append("|---|" + "---:|" * len(LANES))
    for r in runs:
        L.append(f"| `{r['arm']}` | " + " | ".join(f"{r['epochs'][l]:.2f}" for l in LANES) + " |")
    L.append("")

    # ---- the decision the plan pre-committed to
    if "v5_proposed" in names and "v5_agentic_2pct" in names:
        L.append("## The pre-registered agentic decision\n")
        L.append("§8.3 committed to a rule before this run: *if the run with an OOD agentic "
                 "hold-out still favours 2% over 1%, the agentic share rises to 2.0% and the "
                 "synthesis target rises 10B -> 17B.* Here is the paired test.\n")
        L.append("| lane | v5_agentic_2pct - v5_proposed (nats) | 95% CI | P(2% better) |")
        L.append("|---|---:|---|---:|")
        for lane in ["agentic", "reasoning", "code", "indic"]:
            d = paired_lane_delta(runs, "v5_agentic_2pct", "v5_proposed", lane)
            L.append(f"| {lane} | {d['delta']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                     f"{d['p_a_better']:.3f} |")
        ca, cb = ci["v5_agentic_2pct"], ci["v5_proposed"]
        overlap = not (ca["hi"] < cb["lo"] or cb["hi"] < ca["lo"])
        L.append(f"\nW: `v5_agentic_2pct` {ca['mean']:.4f} [{ca['lo']:.4f}, {ca['hi']:.4f}] vs "
                 f"`v5_proposed` {cb['mean']:.4f} [{cb['lo']:.4f}, {cb['hi']:.4f}] - "
                 f"CIs **{'overlap' if overlap else 'do not overlap'}**.\n")

    # ---- the kill condition
    if "naive_web_heavy" in names and "v5_proposed" in names:
        cw, cv = ci["naive_web_heavy"], ci["v5_proposed"]
        fired = not (cw["lo"] > cv["hi"])
        L.append("## The pre-registered kill condition\n")
        L.append(f"*Refutes if the naive web-heavy arm lands within CI of V5-proposed on W.* "
                 f"`naive_web_heavy` {cw['mean']:.4f} [{cw['lo']:.4f}, {cw['hi']:.4f}] vs "
                 f"`v5_proposed` {cv['mean']:.4f} [{cv['lo']:.4f}, {cv['hi']:.4f}] -> "
                 f"**{'FIRED - revert to a high-web mixture' if fired else 'did not fire'}**.\n")

    # ---- the instrument's own resolution, measured rather than assumed
    # seed replicates must come from the SAME regime: a masked report may only calibrate
    # itself against masked seeds. Mixing them compares two different metrics and produces
    # a noise floor an order of magnitude too large.
    stem = a.runs_name[:-5] if a.runs_name.endswith(".json") else a.runs_name
    seed_runs = []
    for p in sorted(RESULTS.glob(f"{stem}_seed*.json")):
        try:
            seed_runs.append(json.loads(p.read_text(encoding="utf-8"))[0])
        except Exception:  # noqa: BLE001
            pass
    if seed_runs:
        ref = next((r for r in runs if r["arm"] == seed_runs[0]["arm"]), None)
        if ref:
            reps = [ref] + seed_runs
            lanes = [l for l in LANES if all(l in r["heldout_nll"] for r in reps)]
            spreads = {l: max(r["heldout_nll"][l] for r in reps)
                       - min(r["heldout_nll"][l] for r in reps) for l in lanes}
            pool = runs + seed_runs
            b2 = {l: min(r["heldout_nll"][l] for r in pool) for l in lanes}
            wof = lambda r: sum(W[l] * r["heldout_nll"][l] / b2[l] for l in lanes if W.get(l))
            wreps = [wof(r) for r in reps]
            L.append("## What this instrument can actually resolve\n")
            L.append(f"`{ref['arm']}` was retrained at {len(reps)} different seeds with the "
                     f"**identical** mixture. Everything that separates those runs is noise, so "
                     f"the spread across them is the smallest difference between two arms that "
                     f"can be believed. A confidence interval over held-out sequences does not "
                     f"capture this - it resamples the test set, not the training run.\n")
            L.append("| lane | spread across seeds (nats) |")
            L.append("|---|---:|")
            for l in sorted(spreads, key=lambda k: -spreads[k]):
                L.append(f"| {l} | {spreads[l]:.4f} |")
            L.append(f"\n**Noise floor on `W`: {max(wreps) - min(wreps):.4f}** "
                     f"({min(wreps):.4f} – {max(wreps):.4f} across seeds).\n")
            others = sorted((abs(wof(r) - wof(ref)), r["arm"]) for r in runs
                            if r["arm"] != ref["arm"])
            nf = max(max(wreps) - min(wreps), 1e-9)
            L.append("| arm | gap to `" + ref["arm"] + "` on W | multiples of the noise floor | verdict |")
            L.append("|---|---:|---:|---|")
            for gap, arm in others:
                m = gap / nf
                L.append(f"| `{arm}` | {gap:.4f} | {m:.1f}x | "
                         f"{'resolvable' if m >= 2 else '**below the noise floor**'} |")
            L.append("")

    L.append("## Per-lane spread\n")
    L.append("| lane | best arm | worst arm | spread (%) |")
    L.append("|---|---|---|---:|")
    for l in LANES:
        vals = [(r["heldout_nll"][l], r["arm"]) for r in runs if l in r["heldout_nll"]]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        L.append(f"| {l} | {lo[1]} ({lo[0]:.3f}) | {hi[1]} ({hi[0]:.3f}) | "
                 f"{100*(hi[0]-lo[0])/lo[0]:.1f}% |")
    L.append("")
    (ROOT / "results" / a.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {ROOT / 'results' / a.out}")


if __name__ == "__main__":
    main()
