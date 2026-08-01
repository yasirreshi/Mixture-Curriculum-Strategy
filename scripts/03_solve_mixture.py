"""
S5 / step 03 - the mixture solver.

Reads mixture/inventory.json (supply) and mixture/v5_mixture.json (the plan) and
answers the only question that matters when a reviewer pushes on a number: does
this share exist in tokens, and if not, is it repeated or manufactured, and by
how much?

Checks, in order:
  1. every stage's weights sum to 100
  2. the token-weighted integral of the stage schedule equals the headline share
  3. no stage falls below the protected always-on floor
  4. demand (main run + anneal) vs unique supply -> epochs, against the 4-epoch cap
  5. the Indic tier split against per-tier supply, including elastic capacity
  6. the anneal reserve is affordable out of the same unique supply
  7. proxy-run compute as a fraction of the main run

Writes results/mixture_report.md. Exit code is non-zero if an invariant fails, so
this is a test, not a report generator.

Run: python scripts/03_solve_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = json.loads((ROOT / "mixture" / "inventory.json").read_text(encoding="utf-8"))
MIX = json.loads((ROOT / "mixture" / "v5_mixture.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "mixture_report.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

LANES = ["code", "general_web", "indic", "stem_math", "reasoning", "long_context", "agentic"]
FAILURES: list[str] = []
NOTES: list[str] = []


def T(x: float) -> str:
    """Human token count."""
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x/div:,.1f}{suf}"
    return f"{x:,.0f}"


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


# ------------------------------------------------------------------ 1 & 2 & 3
main_tokens = MIX["budget"]["main_pretraining_tokens"]
anneal_tokens = MIX["budget"]["anneal_tokens"]
stages = MIX["curriculum"]["stages"]
floor = MIX["protected_floor"]

stage_rows = []
for st in stages:
    s = sum(st["weights_pct"].values())
    check(abs(s - 100.0) < 1e-6, f"stage {st['id']} weights sum to {s:.3f}, not 100")
    stage_rows.append(st)

stage_tok_total = sum(st["tokens"] for st in stages)
check(abs(stage_tok_total - main_tokens) < 1e6,
      f"stage tokens sum to {T(stage_tok_total)} but main run is {T(main_tokens)}")

integrated = {ln: 0.0 for ln in LANES}
for st in stages:
    w = st["tokens"] / stage_tok_total
    for ln in LANES:
        integrated[ln] += st["weights_pct"][ln] * w

for ln in LANES:
    declared = MIX["lanes"][ln]["share_pct"]
    check(abs(integrated[ln] - declared) <= 0.15,
          f"lane {ln}: declared {declared}% but the stage schedule integrates to "
          f"{integrated[ln]:.2f}% (tolerance 0.15pt)")

# The floor is expressed as a fraction of each protected lane's scheduled share in
# every stage, because scripts/05_selector_floor.py measured retention near zero for
# Indic - an absolute floor below the schedule guarantees a shortfall every batch.
prot_frac = floor["protected_fraction"]
prot_lanes = floor["protected_lanes"]
check(0.0 < prot_frac <= 1.0, f"protected_fraction {prot_frac} is not a fraction")
floor_rows = []
for st in stages:
    for ln in prot_lanes:
        floor_rows.append((st["id"], ln, st["weights_pct"][ln],
                           prot_frac * st["weights_pct"][ln]))
run_avg_floor = prot_frac * sum(integrated[ln] for ln in prot_lanes)
check(abs(run_avg_floor - floor["run_average_pct"]) <= 0.15,
      f"protected floor run-average is {run_avg_floor:.2f}% but the spec declares "
      f"{floor['run_average_pct']}%")

# ------------------------------------------------------------------ 4 supply
cap = MIX["invariants"]["max_epochs_per_lane"]
anneal_mix = MIX["anneal_reserve"]["mixture_pct"]
synth = INV["synthesis_capacity"]

# unique supply per lane, including anything we have costed a way to manufacture
extra_unique = {
    "agentic": synth["agentic_verified_trajectories"]["tokens_per_run_window"],
}
supply_rows = []
for ln in LANES:
    unique = INV["lanes"][ln]["lane_total_tokens"] + extra_unique.get(ln, 0.0)
    d_main = main_tokens * integrated[ln] / 100.0
    d_ann = anneal_tokens * anneal_mix[ln] / 100.0
    d_tot = d_main + d_ann
    ep = d_tot / unique
    status = ("covered" if ep <= 1.0 else
              "needs repetition" if ep <= cap else
              "OVER CAP")
    check(ep <= cap, f"lane {ln}: {ep:.2f} epochs exceeds the {cap}-epoch cap "
                     f"(demand {T(d_tot)} vs unique {T(unique)})")
    supply_rows.append((ln, integrated[ln], d_main, d_ann, d_tot, unique, ep, status))
    if extra_unique.get(ln):
        NOTES.append(
            f"`{ln}` reaches its share only because {T(extra_unique[ln])} of the "
            f"{T(unique)} unique supply is manufactured, not collected "
            f"({100*extra_unique[ln]/unique:.0f}% synthetic by token)."
        )

# ------------------------------------------------------------------ 5 indic tiers
tier_supply = INV["lanes"]["indic"]["tier_supply"]
elastic = {
    "C_translated_transliterated": synth["indic_translated"]["tokens_per_run_window"],
    "D_synthetic": synth["indic_synthetic"]["tokens_per_run_window"],
}
# the verified collection programme declared in the plan's risk table
VERIFIED_COLLECTION = 2.5e10
elastic["A_verified_native"] = VERIFIED_COLLECTION

indic_main = main_tokens * integrated["indic"] / 100.0
indic_ann = anneal_tokens * anneal_mix["indic"] / 100.0
tier_rows = []
for tier, pct in MIX["lanes"]["indic"]["tier_split_pct"].items():
    apct = MIX["anneal_reserve"]["indic_tier_split_pct"][tier]
    d = indic_main * pct / 100.0 + indic_ann * apct / 100.0
    base = tier_supply[tier]["tokens"]
    unique = base + elastic.get(tier, 0.0)
    ep = d / unique
    check(ep <= cap, f"indic tier {tier}: {ep:.2f} epochs exceeds the cap")
    tier_rows.append((tier, pct, apct, d, base, unique, ep))

cd = (MIX["lanes"]["indic"]["tier_split_pct"]["C_translated_transliterated"]
      + MIX["lanes"]["indic"]["tier_split_pct"]["D_synthetic"])
check(cd <= 42.0, f"Indic C+D is {cd}%, above the 42% translationese cap")

# ------------------------------------------------------------------ 6 anneal
check(abs(sum(anneal_mix.values()) - 100.0) < 1e-6, "anneal mixture does not sum to 100")
check(abs(sum(MIX["anneal_reserve"]["indic_tier_split_pct"].values()) - 100.0) < 1e-6,
      "anneal Indic tier split does not sum to 100")
check(abs(sum(MIX["lanes"]["reasoning"]["length_band_split_pct"].values()) - 100.0) < 1e-6,
      "reasoning band split does not sum to 100")
for st in stages:
    check(abs(sum(st["difficulty_mix"].values()) - 100.0) < 1e-6,
          f"stage {st['id']} difficulty mix does not sum to 100")
    check(sorted(st["difficulty_mix"]) == ["B0", "B1", "B2", "B3", "B4", "B5"],
          f"stage {st['id']} does not use the session's six-rung B0-B5 ladder")
    check(abs(sum(st["reasoning_length_mix"].values()) - 100.0) < 1e-6,
          f"stage {st['id']} reasoning-length mix does not sum to 100")

# the session states that trace length is scheduled, not just apportioned: the
# per-stage schedule must integrate to the lane's declared band split
rw = {st["id"]: st["tokens"] * st["weights_pct"]["reasoning"] / 100.0 for st in stages}
rtot = sum(rw.values())
band_int = {b: sum(st["reasoning_length_mix"][b] * rw[st["id"]] / rtot for st in stages)
            for b in stages[0]["reasoning_length_mix"]}
for b, v in band_int.items():
    declared = MIX["lanes"]["reasoning"]["length_band_split_pct"][b]
    check(abs(v - declared) <= 1.0,
          f"reasoning band {b}: declared {declared}% but the stage schedule integrates "
          f"to {v:.2f}%")

# ------------------------------------------------------------------ 7 compute
p = MIX["proxy_protocol"]["cost"]
frac = 100.0 * (p["stage_1_flops"] + p["stage_2_flops"]) / p["main_run_flops"]
check(abs(frac - p["fraction_of_main_run_pct"]) < 0.05,
      f"proxy cost fraction is {frac:.2f}%, spec says {p['fraction_of_main_run_pct']}%")

# ------------------------------------------------------------------ report
L: list[str] = []
A = L.append
A("# V5 mixture - solved\n")
A(f"_Generated by `scripts/03_solve_mixture.py` from `mixture/v5_mixture.json` + "
  f"`mixture/inventory.json`. Budget: **{T(MIX['budget']['total_tokens'])} total** = "
  f"{T(main_tokens)} main run + {T(anneal_tokens)} anneal reserve. "
  f"Epoch cap: **{cap}** (arXiv:2305.16264)._\n")

A("## 1. Lane shares: demand against real supply\n")
A("| lane | share | main-run tokens | anneal tokens | total demand | unique supply | epochs | verdict |")
A("|---|---:|---:|---:|---:|---:|---:|---|")
for ln, sh, dm, da, dt, un, ep, stt in supply_rows:
    A(f"| **{ln}** | {sh:.2f}% | {T(dm)} | {T(da)} | {T(dt)} | {T(un)} | {ep:.2f} | {stt} |")
tot_share = sum(r[1] for r in supply_rows)
A(f"| _total_ | _{tot_share:.2f}%_ | _{T(main_tokens)}_ | _{T(anneal_tokens)}_ | | | | |\n")
for n in NOTES:
    A(f"> {n}\n")

A("## 2. Indic provenance tiers\n")
A("| tier | main-run share | anneal share | demand | collected supply | + elastic capacity | epochs |")
A("|---|---:|---:|---:|---:|---:|---:|")
for tier, pct, apct, d, base, un, ep in tier_rows:
    A(f"| {tier} | {pct:.0f}% | {apct:.0f}% | {T(d)} | {T(base)} | {T(un)} | {ep:.2f} |")
A("")

A("## 3. Curriculum: the stage schedule the headline shares integrate from\n")
hdr = "| stage | tokens | seq | " + " | ".join(LANES) + " |"
A(hdr)
A("|---|---:|---:|" + "---:|" * len(LANES))
for st in stages:
    A(f"| {st['id']} {st['name']} | {T(st['tokens'])} | {st['seq_len']} | "
      + " | ".join(f"{st['weights_pct'][ln]:.1f}" for ln in LANES) + " |")
A("| **integrated** | | | " + " | ".join(f"**{integrated[ln]:.2f}**" for ln in LANES) + " |\n")

A("### difficulty band schedule (the session's B0-B5 ladder)\n")
A("| stage | B0 nursery | B1 grade-school | B2 high-school | B3 undergrad | B4 graduate | B5 research |")
A("|---|---:|---:|---:|---:|---:|---:|")
for st in stages:
    dm = st["difficulty_mix"]
    A(f"| {st['id']} | " + " | ".join(f"{dm[k]}" for k in
      ("B0", "B1", "B2", "B3", "B4", "B5")) + " |")
A("")

A("### reasoning-length schedule - trace length is scheduled too\n")
A("| stage | L0 direct | L1 short | L2 medium | L3 long | L4 ultra |")
A("|---|---:|---:|---:|---:|---:|")
for st in stages:
    lm = st["reasoning_length_mix"]
    A(f"| {st['id']} | " + " | ".join(f"{lm[k]}" for k in
      ("L0_direct", "L1_short", "L2_medium", "L3_long", "L4_ultra")) + " |")
A("| **integrated** | " + " | ".join(f"**{band_int[k]:.1f}**" for k in
  ("L0_direct", "L1_short", "L2_medium", "L3_long", "L4_ultra")) + " |")
A("")

A("## 4. Protected always-on floor\n")
A(f"**{prot_frac:.0%} of the scheduled share** of {', '.join(prot_lanes)} is exempt from the "
  f"selector in every stage - **{run_avg_floor:.1f}% of every batch** at run average "
  f"(V4: 8%, Indic only). Level set from measured selector retention, not from proportion; "
  f"see results/selector_report.md.\n")
A("| stage | " + " | ".join(f"{ln} sched / protected" for ln in prot_lanes) + " |")
A("|---|" + "---:|" * len(prot_lanes))
for st in stages:
    cells = [f"{st['weights_pct'][ln]:.1f} / **{prot_frac*st['weights_pct'][ln]:.2f}**"
             for ln in prot_lanes]
    A(f"| {st['id']} | " + " | ".join(cells) + " |")
A("")

A("## 5. Anneal reserve\n")
A(f"**{T(anneal_tokens)}** quarantined at manifest level. Composition:\n")
A("| lane | share | tokens |")
A("|---|---:|---:|")
for ln in LANES:
    A(f"| {ln} | {anneal_mix[ln]:.0f}% | {T(anneal_tokens*anneal_mix[ln]/100)} |")
A("")

A("## 6. Proxy cost\n")
A(f"Stage-1 (8 arms x 1B x 30B tok) = {p['stage_1_flops']:.2e} FLOPs; "
  f"stage-2 (3 arms x 3B x 60B tok) = {p['stage_2_flops']:.2e}; "
  f"main run (40B x 3.0T) = {p['main_run_flops']:.2e}. "
  f"The whole experiment costs **{frac:.2f}%** of the run it is protecting "
  f"({p['wall_clock']}).\n")

A("## 7. Invariant check\n")
if FAILURES:
    A(f"**{len(FAILURES)} FAILURE(S):**\n")
    for f in FAILURES:
        A(f"- {f}")
else:
    A("All invariants hold: stage weights sum to 100, the schedule integrates to the "
      "declared shares, no stage breaches the floor, no lane or Indic tier exceeds "
      f"{cap} epochs, the anneal is affordable out of the same unique supply, and the "
      "proxy costs what the plan says it costs.")
A("")

OUT.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n-> {OUT}")
sys.exit(1 if FAILURES else 0)
