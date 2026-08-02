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

# ---------------------------------------------------------- 5b reasoning length bands
# The Indic lane is priced tier by tier because the constraint lives in the tiers, not the
# headline. The same is true of reasoning: L3/L4 have ZERO collected supply, so the bands
# have to clear the epoch cap against the distillation programme that funds them, not
# against the lane total. Without this check the plan's own principle is applied to one
# lane and not the other.
lb_split = MIX["lanes"]["reasoning"]["length_band_split_pct"]
lb_ann = MIX["anneal_reserve"]["reasoning_band_split_pct"]
reasoning_main = main_tokens * integrated["reasoning"] / 100.0
reasoning_ann = anneal_tokens * anneal_mix["reasoning"] / 100.0
LONG_BANDS = ("L3_long", "L4_ultra")
long_demand = (reasoning_main * sum(lb_split[b] for b in LONG_BANDS) / 100.0
               + reasoning_ann * sum(lb_ann.get(b, 0) for b in LONG_BANDS) / 100.0)
long_supply = synth["reasoning_long_traces_L3_L4"]["tokens_per_run_window"]
long_ep = long_demand / long_supply
check(long_ep <= cap,
      f"reasoning L3+L4: {long_ep:.2f} epochs against the {T(long_supply)} distillation "
      f"programme, over the {cap}-epoch cap")
short_demand = (reasoning_main * sum(lb_split[b] for b in lb_split if b not in LONG_BANDS) / 100.0
                + reasoning_ann * sum(v for k, v in lb_ann.items() if k not in LONG_BANDS) / 100.0)
short_supply = INV["lanes"]["reasoning"]["lane_total_tokens"]
short_ep = short_demand / short_supply
check(short_ep <= cap, f"reasoning L0-L2: {short_ep:.2f} epochs, over the cap")

# ------------------------------------- 5b2 the session's five-stage view is the same plan
# The session names five stages (Seed -> General -> Reasoning -> Long-context -> Anneal); the
# internal schedule is A/B/C/D. `session_stage_view` re-expresses one as the other. It is only
# trustworthy if it is a re-expression rather than a second set of numbers, so integrate it and
# require the identical lane shares.
ssv = MIX["session_stage_view"]["stages"]
pre_stages = [s for s in ssv if s["id"] != "anneal"]
ssv_tokens = sum(s["tokens"] for s in pre_stages)
check(abs(ssv_tokens - main_tokens) < 1e6,
      f"session-stage view covers {T(ssv_tokens)} of pretraining, not {T(main_tokens)}")
for s in ssv:
    tot = sum(s["weights_pct"].values())
    check(abs(tot - 100.0) < 0.02, f"session stage {s['id']} weights sum to {tot:.3f}, not 100")
for ln in LANES:
    integ_ssv = sum(s["tokens"] * s["weights_pct"][ln] for s in pre_stages) / ssv_tokens
    check(abs(integ_ssv - integrated[ln]) <= 0.02,
          f"session-stage view integrates {ln} to {integ_ssv:.3f}%, but the A/B/C/D schedule "
          f"gives {integrated[ln]:.3f}% - the two views have diverged")
# the anneal row of the five-stage view must be the declared reserve mixture, not a copy that drifts
for ln in LANES:
    a_ssv = next(s for s in ssv if s["id"] == "anneal")["weights_pct"][ln]
    check(abs(a_ssv - anneal_mix[ln]) < 1e-6,
          f"session-stage view's anneal {ln} is {a_ssv}, reserve declares {anneal_mix[ln]}")

# ------------------------------------------------- 5b3 the whole-lifecycle budget
lc = MIX["training_lifecycle"]["stages"]
lc_total = sum(s["tokens"] for s in lc)
lc_pre = next(s for s in lc if s["id"] == "pretraining")["tokens"]
lc_mid = next(s for s in lc if s["id"] == "midtraining")["tokens"]
check(abs(lc_pre - main_tokens) < 1e6,
      "lifecycle pretraining row must equal the main-run budget")
check(abs(lc_mid - anneal_tokens) < 1e6,
      "lifecycle mid-training row must equal the declared anneal reserve")
pre_share = 100.0 * lc_pre / lc_total
check(93.0 <= pre_share <= 97.0,
      f"pretraining is {pre_share:.1f}% of all training tokens; the session's lifecycle panel "
      f"puts it at ~95%, so anything outside 93-97% needs saying out loud")

# ------------------------------------------------- 5c reasoning quality-gate stress test
# The lane's 85B is a GROSS figure. Cleaning measured that a verified-step gate drops 75%
# of the only component carrying step-level human labels. That measurement is not carried
# into the headline epoch count anywhere else, so it is carried here: what survival rate
# does the share actually require, and how does that compare to the one rate we measured?
qg = INV["lanes"]["reasoning"]["quality_gate"]
reasoning_demand = reasoning_main + reasoning_ann
break_even = reasoning_demand / (cap * short_supply)
check(abs(break_even - qg["break_even_survival"]) < 0.005,
      f"reasoning break-even survival recomputes to {break_even:.3f}, but the inventory "
      f"declares {qg['break_even_survival']}")
check("gate" in qg and qg["measured_survival_prm800k"] < break_even,
      "the reasoning quality gate must state a delivery gate whenever measured survival "
      "sits below the break-even the share requires")

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
A("| tier | main-run share | anneal share | demand | collected today | fundable supply | epochs |")
A("|---|---:|---:|---:|---:|---:|---:|")
for tier, pct, apct, d, base, un, ep in tier_rows:
    A(f"| {tier} | {pct:.0f}% | {apct:.0f}% | {T(d)} | {T(base)} | {T(un)} | {ep:.2f} |")
A("")
A(f"> **Two different supply bases, stated so they are not read as one.** The lane row in "
  f"§1 prices Indic against **collected** supply only ({T(sum(r[4] for r in tier_rows))}, the "
  f"course inventory's lane total) and reports {(indic_main + indic_ann) / sum(r[4] for r in tier_rows):.2f} "
  f"epochs. The tier rows here price each tier against **fundable** supply - collected plus "
  f"the collection programme for Tier A and the costed translation/generation capacity for "
  f"C and D ({T(sum(r[5] for r in tier_rows))} in total) - which is why the per-tier epochs "
  f"are lower. The lane-level figure is the conservative one and is the number quoted in the "
  f"headline table; the tier figures are what each tier's own funding plan has to deliver.\n")

A("## 2b. Reasoning: the constraint lives in the length bands, not the lane\n")
A("| band group | demand | supply it is drawn from | epochs |")
A("|---|---:|---|---:|")
A(f"| L0-L2 (collected) | {T(short_demand)} | {T(short_supply)} collected open corpus | {short_ep:.2f} |")
A(f"| **L3+L4 (zero collected)** | {T(long_demand)} | {T(long_supply)} distillation programme | "
  f"**{long_ep:.2f}** |")
A("")
A(f"> Cleaning found **zero** collected traces at L3 or above, so the long half of this lane "
  f"is drawn entirely against the {T(long_supply)} distillation line item and has to clear the "
  f"cap on its own. It does, at {long_ep:.2f} epochs. Pricing the lane only at lane level "
  f"would have hidden that the two halves are funded from completely different places.\n")

A("### the quality-gate stress test this lane's own measurement forces\n")
A(f"The {T(short_supply)} is a **gross** figure. Cleaning measured that a verified-step gate "
  f"drops **{100*(1-qg['measured_survival_prm800k']):.1f}%** of PRM800K - the only component of "
  f"the lane carrying step-level human labels. That measurement is priced here rather than "
  f"noted and forgotten.\n")
A("| survival of the nominal supply | gate-surviving supply | epochs | verdict |")
A("|---|---:|---:|---|")
for s in (qg["measured_survival_prm800k"], 0.50, break_even, 0.80, 1.00):
    sup = short_supply * s
    e = reasoning_demand / sup
    tag = ("**measured on PRM800K**" if abs(s - qg["measured_survival_prm800k"]) < 1e-6 else
           "**break-even**" if abs(s - break_even) < 5e-3 else
           "as planned" if s == 1.0 else "")
    A(f"| {100*s:.1f}% {tag} | {T(sup)} | {e:.2f} | {'inside cap' if e <= cap else '**OVER CAP**'} |")
A("")
A(f"> The reasoning share of {integrated['reasoning']:.2f}% is fundable **only if at least "
  f"{100*break_even:.1f}%** of the nominal {T(short_supply)} survives quality gating. The one "
  f"component where survival could be measured rather than assumed survived at "
  f"**{100*qg['measured_survival_prm800k']:.1f}%**. The other 78B is answer-verified rather "
  f"than step-verified, so PRM800K's rate is a worst case and is not assumed lane-wide - but "
  f"the gap is wide enough that the share cannot be called settled. Gate: "
  f"{qg['gate']}\n")

A("## 2c. The session's five stages — the same schedule, in the session's own vocabulary\n")
A("| session stage | tokens | seq | " + " | ".join(LANES) + " |")
A("|---|---:|---:|" + "---:|" * len(LANES))
for s in ssv:
    A(f"| **{s['name']}** | {T(s['tokens'])} | {s['seq_len']} | "
      + " | ".join(f"{s['weights_pct'][ln]:.1f}" for ln in LANES) + " |")
A("| _integrated (pretraining only)_ | _" + T(ssv_tokens) + "_ | | "
  + " | ".join(f"**{integrated[ln]:.2f}**" for ln in LANES) + " |\n")
A(f"> {MIX['session_stage_view']['mapping']} This is a **re-expression, not a second set of "
  f"numbers**: the four pretraining rows integrate to the identical lane shares, and this script "
  f"fails if they ever diverge.\n")

A("## 2d. Where this session sits in the whole training lifecycle\n")
A("| stage | tokens | share of all training data | loss signal | owned by |")
A("|---|---:|---:|---|---|")
for s in lc:
    A(f"| **{s['name']}** | {T(s['tokens']) if s['tokens'] else '—'} | "
      f"{100*s['tokens']/lc_total:.2f}% | {s['loss']} | {s['owned_by']} |")
A(f"| _total_ | _{T(lc_total)}_ | | | |\n")
A(f"> Pretraining is **{pre_share:.1f}%** of all training tokens, against the session's ~95%. "
  f"Every later stage is comparatively tiny — which is exactly why the mixture decided here is "
  f"the consequential one. The reserve's 18B of L3/L4 reasoning is the feedstock for reasoning "
  f"training in Sessions 17–18: a data decision made now determines what is possible then.\n")

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
