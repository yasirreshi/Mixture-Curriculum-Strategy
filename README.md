# V5 data mixture & curriculum — Session 5

The mixture-and-curriculum plan for the 40B India-first model: **what it sees, how much of it, and in
what order.** Every share is priced against real supply, and every one was allowed to be wrong — ten
were.

📊 **[Open the visual brief →](https://yasirreshi.github.io/Mixture-Curriculum-Strategy/walkthrough.html)**
— the whole plan as charts. Start there. *(source: [`walkthrough.html`](walkthrough.html))*
🔬 **[results/](results/)** — the generated reports every number in this repo comes from.
✅ **Reproduce or check it:** `python scripts/03_solve_mixture.py` (the arithmetic) ·
`python scripts/12_assignment_audit.py` (the brief) — both exit non-zero on failure.

---

## Summary — the whole thing in one page

**The allocation.** A 3.0T budget across seven lanes: general web 31.03 · code 24.00 · Indic 17.90 ·
STEM 11.52 · long-context 7.56 · reasoning 7.03 · **agentic 0.96**. Not round numbers, because they
are the *integral of a stage schedule* — the curriculum is the primary object and the share falls out
of it. The run is staged **Seed → General → Reasoning → Long-context → Anneal**, across which web
collapses 70 → 6 and agentic climbs 0 → 10.

**The one rule that makes it defensible.** No lane may exceed **4 epochs** over its unique supply, and
the rule is applied *asymmetrically*: a lane that fails loses share, and the cap is never raised. That
asymmetry is the only thing separating this from a spreadsheet of round numbers — and it survived its
hardest test when evidence arrived *favouring* a bigger agentic share and the answer was to fund more
supply rather than lift the cap.

**What stands behind it.** 12 scripts, 31 licensed sources, **191.4M cumulative cleaned tokens**, and
**23 proxy models trained** — the largest 42.1M params × 28.0M tokens at context 1024, epoch-honest,
with out-of-distribution hold-outs for the three lanes where in-distribution scoring would reward
format memorisation rather than capability.

**Four decisions carry the plan. Each is forced by arithmetic, not chosen by preference:**

| decision | the calculation that forces it |
|---|---|
| **Agentic 0.96%** — deliberately under the session's own ≥2% floor | 2% spends 68B against 10.63B of supply = **6.4 epochs**. The floor as printed cannot be met by collected data at any run size the session considers |
| **Indic Tier A 28%**, where the composer sets 40% | 40% is 208B against ~85B of verified supply = **2.4 epochs in the main run alone**. The lane total was never the constraint — the verified tier is |
| **Protected floor 23.3%**, against V4's 8% | Measured Indic retention under an English-and-code selector is **0.0%**. A floor set below the schedule delivers exactly itself, so it has to be a *fraction of* the schedule |
| **Anneal reserve 100B**, quarantined before the run starts | Sized by what can be **admitted**, not by a percentage: the two lanes this model exists for fill at **0.6% and 0.0%**. The reserve's hard problem is manufacturing, not selection |

**What the experiments established.** Four runs, and the method worked in the way that matters — it
produced conclusions the author did not choose:

- **The mixture is a capability decision, measurably.** `naive_web_heavy` came last on all three
  instruments — on the last by **34.8× the measured noise floor**. The one pre-registered kill
  condition never fired. "Crawl what is cheap" is not defensible.
- **The instrument knows its own resolution.** Retraining one mixture at three seeds puts the noise
  floor at **0.0022** on `W`. Four of six arm comparisons clear it; two do not, and are reported as
  *unresolvable* rather than as results.
- **Two findings went against the plan, and both are reported.** It placed **4th of 6**, then **last
  of 4**, on the CPU proxies; and under an OOD hold-out with the plan's own loss mask, **2% agentic
  beat the proposed 0.96% on three seeds of three, with disjoint ranges**. Each produced a correction
  to the protocol or a funded change to supply — never a rewrite of the criterion. The agentic share
  waits for the 1B run its rule names, but the **supply is funded to 17B now**, because supply has
  lead time and a share does not.
- **And one finding neither run was looking for:** masked and unmasked scoring disagree about which
  mixture is better *on identical data*. **The loss mask is not a training detail — it decides the
  answer**, which is now a required correction to the 1B protocol.

**What is still open.** The reasoning lane's 85B is a *gross* figure: our own cleaning measured a 75.5%
quality-gate drop against it, and the share needs **65.3% survival** to be fundable while the one
component we could measure survived at **24.5%**. That is the largest unpriced risk here, and it is
priced as a stress test with a sampling gate rather than defended.

**How to check any of it.** Two scripts are tests, not reports:
[`03_solve_mixture.py`](scripts/03_solve_mixture.py) checks the plan against its own arithmetic and
[`12_assignment_audit.py`](scripts/12_assignment_audit.py) checks it against the brief — **56/56**.
Both exit non-zero on failure, so "this plan is consistent" is a test result rather than a claim.

| | | why |
|---|---|---|
| Budget | **3.0T** = 2.9T main run + **100B anneal reserve** | Session sets 2.4–4T. 3.0T on 40B is 75 tok/param — deliberately over-trained, because a served model pays at inference |
| Hard rule | **≤ 4 epochs per lane**, asymmetric — a failing lane loses share, the cap never rises | Muennighoff: ≤4 passes ≈ fresh data at fixed compute; past that, added compute decays to zero. The asymmetry is what stops a share being a wish |
| Protected floor | **23.3% of every batch** (Indic + agentic + reasoning) | Not chosen — *derived*. Measured Indic retention under an English-and-code selector is **0.0%**, so the floor must be a fraction of the schedule, not an absolute below it |
| Anneal reserve | **100B**, quarantined before the run starts | Sized by what can be **admitted**, not by a percentage: the two lanes we most want fill at 0.6% and 0.0% |
| Data gate | **191.4M** cumulative cleaned tokens, 31 licensed sources | The brief reviews nothing until the gate is met. S4 63.1M + S5 128.3M |
| Proxy | 1B/3B pre-registered · **23 models actually trained** | A data decision is a hypothesis until a cheap experiment has tested it. Cost: 0.65% of the run it protects |
| Self-checks | solver **passes** · assignment audit **56/56** | Both exit non-zero on failure, so "the plan is consistent" is a test result rather than a claim |

---

## The mixture

| lane | share | demand ÷ supply = epochs | why this number and not another | buys |
|---|---:|---|---|---|
| general web | **31.03%** | 906B ÷ 4.50T = **0.20** | RegMix finds web, not curated sets, correlates best downstream — so cutting it hard is a real risk, not a free win. Faded 56→11.5 rather than cut. **Weakest number here:** my own proxy measures it as the cheapest large lane to cut (−0.214) | MMLU, ARC, TriviaQA |
| code | **24.00%** | 714B ÷ 1.10T = **0.65** | Target capability #1 and the only large lane where supply never binds — every token fresh. V4 ended at 35%; cut to 24% because the model also carries Indic duty, and a code-saturated model programs anything with no common sense | LiveCodeBench, Aider, HumanEval+ |
| Indic | **17.90%** | 541B ÷ 276B = **1.96** | Not 25%: the *lane* would fit, the **verified tier** would not. At 25% Tier A needs >4 epochs or dilution with translated text until the model learns translationese. The constraint lives one level down — see the tier table | MILU, IndicGenBench, FLORES-200 |
| STEM / math | **11.52%** | 348B ÷ 250B = **1.39** | V4 ended at 39%. Sustained here that is 1.13T ÷ 250B = **4.5 epochs, over the cap** — so it ramps 6→14 and holds, and the scarce top (proofs, olympiad) is pushed into the anneal rather than burned early | GPQA, MMLU-Pro STEM |
| long context | **7.56%** | 231B ÷ 100B = **2.31** | Not a lane you sprinkle: batches must be length-homogeneous, so a short sample in a long batch is wasted compute. Concentrated 2.5/3.0/16.8/13.0 by stage, almost all after the model can already read | RULER, long-eval, SWE-bench context |
| reasoning | **7.03%** | 222B ÷ 85B = **2.61** | Inside the cap with **no slack**, which is why it is protected. Pretraining installs the *structure* of a worked argument; the capability is bought in RLVR. ⚠ The 85B is **gross** — needs 65.3% quality-gate survival to hold | AIME, MATH-500, HLE |
| agentic | **0.96%** | 37.9B ÷ 10.6B = **3.57** | The composer's 2% would spend 68B ÷ 10.63B = **6.4 epochs**, over the cap. 0.96% is the largest share the supply funds — and 10.0B of that 10.6B is **manufactured**, costed per environment family | SWE-bench, BFCL, tau2-bench |

Shares are the token-weighted integral of the stage schedule — the curriculum is the primary object.
**Only agentic runs near the cap**, and only because 10.0B of its 10.6B supply is manufactured. It is
the one lane funded by data that does not yet exist, and that is said where the number is.

**Indic, tier by tier** — because the lane total was never the constraint:

| tier | main run | anneal | demand ÷ fundable = epochs | why this share |
|---|---:|---:|---|---|
| **A** verified native | **28%** | **70%** | 160.7B ÷ 85B = **1.89** | The session preset is 40%. That is 0.40 × 519B = **208B against ~85B of supply even after the collection programme = 2.4 epochs in the main run alone**, before the anneal takes its cut. It does not fit. So 28% here — and the anneal **inverts to 70%**, spending the best tokens at low LR on a model that can use them |
| **B** unverified crawl | 30% | 5% | 156.8B ÷ 135B = **1.16** | The shock absorber. It has the most headroom of any tier, so if Tier-A collection lands under 70B **this is what absorbs the shortfall** — never C or D |
| **C** translated | 24% | 10% | 126.8B ÷ 125B = **1.01** | Elastic — we can manufacture more at translation cost — which is exactly why it needs a **ceiling, not a floor**. At 1.01 epochs its capacity is *fully consumed*: no slack to absorb anything |
| **D** synthetic | 18% | 15% | 96.7B ÷ 96B = **1.01** | Also elastic, and capped harder because synthetic Indic inherits one teacher's register and flattens dialect variety. `C + D ≤ 42%` is a hard solver constraint |

Two supply bases, so they are not read as one: the lane row above prices Indic against **collected**
supply (276B → 1.96 epochs); these tier rows price against **fundable** supply — collected plus the
Tier-A collection programme plus costed translation/generation capacity (441B). The lane figure is the
conservative one and is the one quoted in the headline.

**What actually fills each lane**, from the inventory. Every line in
[`inventory.json`](mixture/inventory.json) carries an evidence tag — `[course]`, `[paper]`,
`[estimate]` or `[measured]` — so a reviewer can see which numbers are counted and which are derived:

| lane | the datasets behind the number | what makes it fundable, or not |
|---|---|---|
| general web | DCLM-baseline 2.6T · FineWeb-Edu 1.3T · curated reference (Wikipedia 320+ langs, PD books) 600B | Vastly over-supplied at 0.20 epochs. The only lane where the question is what to *exclude* |
| code | The Stack v2 permissive 900B · GitHub PRs & issues 80B · notebooks & docs 60B · **commit diffs 40B** | Diffs are listed separately on purpose: a diff is the training *shape* that matches Aider's metric, so it is a lane requirement rather than a nice-to-have |
| Indic | Sangraha 251B · IndicCorp v2 20.9B · samvaad-hi (S4, cleaned) · Indic Wikipedia ×12 (cleaned here) | Sangraha's 251B is explicitly a *blend* of verified, unverified and synthetic — which is exactly why this lane is priced by tier, not by total |
| STEM / math | Nemotron-CC-Math 130B · proof-pile-2 / OpenWebMath 55B · FineMath 34B · arXiv full text 30B | Adequate in volume; the binding limit is the *top* of the distribution (proofs, olympiad), which is why that fraction is pushed into the anneal |
| long context | repo-level packs ~40B · books ~25B · arXiv, court judgments & gazettes ~25B · multi-doc synthetic ~10B | Packs are the majority because they match SWE-bench's shape — but **packed tokens are not new unique tokens** and are charged to the source lane |
| reasoning | OpenThoughts-3 / OpenR1 / OpenMathReasoning ~40B · Nemotron splits ~30B · NuminaMath ~8B · PRM800K 1B · GSM8K | The only lane whose headline supply we actively distrust: 85B is gross, and the L3/L4 half has **zero** collected tokens |
| agentic | ToolBench 80M · Gorilla + BFCL 90M · xLAM 60M · SWE-Gym ~200M · terminal/OS/web traces ~200M | **0.63B total — two orders of magnitude below every other lane.** The number that forces the whole synthesis programme |

---

## The split across the session's five stages

Seed → General → Reasoning → Long-context → Anneal. Every lane, every stage:

| session stage | web | code | Indic | STEM | reason | long-ctx | agentic | why the stage has this shape |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Seed** <br>60B · 2K | 70.0 | 6.0 | 20.0 | 3.0 | 1.0 | 0.0 | 0.0 | Bands B0/B1 only. **Indic already at 20%** — its highest until the anneal — because script and morphology must be learned while the embedding table is still plastic. **Agentic 0:** a format lesson is wasted on a model that cannot yet read |
| **General** <br>840B · 4K | 55.0 | 18.9 | 14.6 | 6.2 | 2.1 | 2.7 | 0.5 | The broad base. Web peaks and falls for the rest of the run. Every lane is present, so nothing is introduced cold later — the V4 failure mode |
| **Reasoning** <br>900B · 8K | 28.0 | 30.0 | 18.0 | 14.0 | 6.0 | 3.0 | 1.0 | **Code overtakes web** (30 vs 28), STEM more than doubles, reasoning triples. This is the session's "code + logic" stage |
| **Long-context** <br>1.1T · 32K→128K | 13.1 | 24.0 | 20.2 | 14.0 | 12.0 | 15.4 | 1.3 | Context stretches only now, with length-homogeneous batches and RoPE-θ extension on long data only. **Indic peaks** (24% by the end): cross-lingual transfer is cheapest from an already-strong technical model |
| **Anneal** <br>100B · low LR | 6.0 | 18.0 | 22.0 | 14.0 | 18.0 | 12.0 | **10.0** | The inversion. **Agentic ×7.7, reasoning ×1.5**, Indic at 70% Tier-A verified, web collapses to 6. The scarcest material is spent last, on a model able to use it |
| *integrated* <br>*pretraining 2.9T* | **31.03** | **24.00** | **17.90** | **11.52** | **7.03** | **7.56** | **0.96** | *These are the headline shares — derived from the schedule above, not chosen independently of it* |

Read the columns downward. **Web collapses 70 → 6.** **Code peaks in the Reasoning stage** (30%,
overtaking web) then holds. **Indic never drops below 14.6%** and peaks in the anneal at 70% Tier-A
verified. **Agentic starts at literally zero** — it is a format lesson that needs a model able to read
first — and ends at 10×, the biggest inversion in the plan.

This is a **re-expression, not a second set of numbers**. The internal schedule is A/B/C/D; Seed is
A0, General is the rest of A, Reasoning is B, Long-context is C+D. The solver integrates it and fails
if the two views ever diverge.

Two ladders are scheduled independently across these stages: **difficulty B0–B5** and **trace length
L0–L4**, each with a real document from `data/clean/` at every rung. L4 is reported **empty** rather
than filled.

---

## Where this session sits in the whole lifecycle

The session's timeline is pretraining → mid-training/anneal → SFT → reasoning training → preference
alignment → serving, and it states pretraining is ~95% of all training data. Ours:

| stage | tokens | share | loss signal | why this size | owned by |
|---|---:|---:|---|---|---|
| **Pretraining** | 2.9T | **95.55%** | next-token on *every* token | The session puts pretraining at ~95%; we land at 95.55%. At 40B params this is 75 tok/param — see the scaling note below | **Session 5 — this doc** |
| **Mid-training / anneal** | 100B | 3.29% | next-token, low LR | Session says ~2%. We run larger and justify it by **what can be admitted**: at the measured fill rates a smaller reserve could not be filled at Tier-A quality without padding it with second-grade data | **Session 5 — this doc** |
| Supervised fine-tuning | 20B | 0.66% | masked to the assistant response | Sized from the agentic argument: SWE-bench is won here, and **10–20B is a large SFT budget** rather than a rounding error — which is precisely why 0.96% suffices in pretraining | later sessions |
| Reasoning training (RLVR) | 12B | 0.40% | masked response **+ verifier reward** | Small in tokens because the signal is a *reward*, not tokens. **Fed by the reserve's 18B of L3/L4** — which is why trace length is banded now | Sessions 17–18 |
| Preference alignment | 3B | 0.10% | reward only, no token targets | Preference pairs are expensive per example and few are needed; the signal carries no token targets at all | later sessions |
| Serving | — | — | none | No training signal. The effort dial **selects among behaviours installed in stages 1–4** — it cannot create them | deployment |

**Why this table belongs in a mixture plan.** Three things fall out of it:

1. **98.8% of all training tokens are decided here.** Every later stage is a rounding error by volume,
   which is why getting this mixture wrong cannot be fixed downstream.
2. **The loss signal narrows at every step** — every token → response only → response + reward →
   reward alone → nothing. That is the same principle as §4's masking convention applied *within*
   pretraining, and §7 shows it deciding an experimental result.
3. **The reserve's 18B of L3/L4 reasoning is stage 4's feedstock.** A data decision made now
   determines what is possible in Sessions 17–18 — which is precisely why trace length is banded now
   rather than later, and why L4 being empty is a problem to solve today.

Pretraining installs the **format** of reasoning and tool use. The **capability** is bought later, on
data this session is responsible for reserving.

---

## What changed since V4, and why

Every figure in the V4 column is from the session's own tie-in panel — this is a delta against a real
previous run, not against a hypothetical.

| dimension | V4 | V5 | why it changed |
|---|---|---|---|
| web schedule | faded **72 → 18** | faded **56 → 11.5** | Starts lower and ends lower, because V5 carries Indic and agentic duty V4 did not. It does not go lower still only because RegMix warns web correlates best downstream |
| code schedule | ramped **13 → 35** | ramped **18 → 30** | Starts *higher* — code is present from stage A rather than introduced late — and ends lower, because a code-saturated model programs anything and has no common sense |
| STEM schedule | ramped **7 → 39** | ramped **6 → 14** | V4's 39% sustained across our run is 1.13T ÷ 250B = **4.5 epochs, over the cap**. The scarce top of the distribution goes to the anneal instead of being burned early |
| protected floor | **8%**, Indic only | **23.3%**, Indic + agentic + reasoning | The session extends protection to three lanes. The *level* is not inherited — it is derived from measured retention, where Indic scored **0.0%** |
| agentic | *"V4 had almost none of it"* | **0.96%** + a 10B costed synthesis programme | V4 was not ready to look at the lane. V5 sizes it against real supply (0.63B) rather than against intent — which is why it lands under the session's own 2% floor |
| Indic | inside the 8% always-on lane | **17.9%**, split across four provenance tiers | A single headline number hides the only binding constraint. The lane total was never the problem; Tier A is |
| anneal reserve | not held back | **100B**, quarantined at manifest level before the run | If the selector eats the best Indic, agentic and reasoning data early, there is nothing special left for the cooldown. So it is an allocation decision now, not an end-of-run discovery |
| stage transitions | step change → **gradient norm jumped ~150×** when a raised Hindi share met frozen embeddings | **8B-token blended warm-up**, rollback at >3× over 2B tokens | V4's incident is the direct evidence. This is the one change made purely to avoid repeating a known failure |

---

## How I built it

Twelve steps, each one a script; `results/` holds what each produced.

| | step | what it did | what came back |
|---|---|---|---|
| 01 | [`01_fetch_lanes.py`](scripts/01_fetch_lanes.py) | acquire the four starved lanes | 30 sources, provenance logged per file |
| 02 | [`02_clean_lanes.py`](scripts/02_clean_lanes.py) | 8-stage clean + loss masks + bands + reserve flags | 128.0M tokens, 12 shards, per-shard hashes |
| 03 | [`03_solve_mixture.py`](scripts/03_solve_mixture.py) | **test:** the plan against its own arithmetic | two dozen invariants, non-zero exit on failure |
| 04 | [`04_proxy_ablation.py`](scripts/04_proxy_ablation.py) | 6-arm CPU ablation | web-heavy last by 16%; **my plan 4th of 6** |
| 05 | [`05_selector_floor.py`](scripts/05_selector_floor.py) | measure what an English-and-code selector does | **Indic retention 0.0%** → rewrote the floor |
| 06–07 | [`06`](scripts/06_band_examples.py) · [`07`](scripts/07_cleaning_report.py) | pull a real example per band; account the gate | B0–B5 + L0–L3 with measured token counts |
| 08 | [`08_proxy_analysis.py`](scripts/08_proxy_analysis.py) | fit share → loss per lane | web is the flattest large lane (−0.214) |
| 09 | [`09_build_long_traces.py`](scripts/09_build_long_traces.py) | build the L3 band from PRM800K's rejected branches | **253 real L3 traces**; L4 still empty |
| 10 | [`10_gpu_proxy.py`](scripts/10_gpu_proxy.py) | GPU proxy: supply caps + **OOD hold-outs** + masking | the agentic decision, settled — see below |
| 11 | [`11_fetch_more.py`](scripts/11_fetch_more.py) | collect for the lanes that starve the *experiment* | epoch-honest ceiling 2.26M → **30.2M** |
| 12 | [`12_assignment_audit.py`](scripts/12_assignment_audit.py) | **test:** the plan against the brief | **56/56**; caught 253 mis-banded documents |

```bash
pip install -r requirements.txt
python scripts/01_fetch_lanes.py && python scripts/02_clean_lanes.py
python scripts/03_solve_mixture.py        # TEST 1 — arithmetic
python scripts/10_gpu_proxy.py --tokens 28000000   # needs CUDA
python scripts/12_assignment_audit.py     # TEST 2 — the brief
```

---

## What the experiments found

Four runs, 23 models, largest 42.1M params × 28.0M tokens at context 1024 — epoch-honest, with
out-of-distribution hold-outs. Two of the four returned a verdict against the plan, which is the
clearest evidence the method was working.

**The kill condition never fired.** `naive_web_heavy` came last on all three instruments — on the
last, by **34.8× the measured noise floor**. "Crawl what is cheap" is not defensible.

**The instrument knows its own resolution.** Retraining one mixture at three seeds puts the noise floor
at **0.0022** on `W`. Four of six arm comparisons clear it; two do not, and are reported as
unresolvable rather than as results.

**The plan lost the argument about its own most-defended number.** With an out-of-distribution agentic
hold-out and the plan's own loss-masking convention, the 2% arm beat the 0.96% arm on **three seeds of
three, with disjoint ranges** (mean −0.0917 nats) — while paying 5.46 epochs against 2.73.

> The headline share still does not move, because the pre-registered rule names *the 1B run* and this
> is 42M params — the same standard that kept `code_heavy` out of the table when the proxy favoured it.
> **What does move is the supply:** the agentic synthesis programme is funded to **17B now**, not on
> confirmation, because supply has lead time and a share does not.

**And the finding underneath it:** masked and unmasked runs disagree about which mixture is better on
identical data. **The loss mask is not a training detail — it decides the answer.**

---

## The technical parameters this mixture assumes

Everything above is a *token allocation*. It only means something against the run it feeds, so the
assumptions are stated rather than left implicit.

**Compute, and why 3.0T for a 40B model.** 6ND = **7.2e23 FLOPs**. At 3.0T tokens on 40B parameters
that is **75 tokens per parameter — 3.75× past Chinchilla-optimal** (~20:1). This is deliberate and
it is the kind of number a reviewer should challenge, so: compute-optimal would say either *150B
params on 3.0T* or *40B on 800B*. Both are worse for this model. A model serving Indian users at scale
pays its real cost at **inference**, not training, and a 40B trained long is far cheaper to serve than
a 150B trained short. Over-training is the price of a servable model — and it also means **repetition
discipline matters more here, not less**, because 75:1 leaves no room to pad a lane with a second pass.

**Tokenizer.** 200k vocab, tied embeddings (carried from Session 3). The vocabulary is not incidental
to this mixture: at our measured fertility, one Indic word costs **2–4× an English one** (1.98 Urdu →
7.48 Malayalam against 1.81 English, measured on the 32k proxy). A "17.9% Indic share" therefore buys
noticeably less *language* than it buys tokens, and the 200k vocab exists to close that gap. Any
change to the tokenizer changes what this share is worth.

**Packing and batching.** Batches are **length-homogeneous** — a 4K and a 32K sample never share a
batch, because a short sample in a long batch is wasted compute. Documents are packed to the sequence
length with **attention masked at document boundaries**, so no sample attends across an unrelated
document. Repo-level packs are the exception by design: they *are* one document.

**What the pipeline guarantees, stage by stage** (all measured, `results/cleaning_report.md`):

| stage | method | this pass | why it is set this way |
|---|---|---|---|
| quality filter | length, symbol ratio, boilerplate | 99.51% survival | High survival is expected — the sources are already curated. A low rate here would mean the *acquisition* was wrong, not the filter |
| language ID | runtime script check against the **declared** language | 165 caught | Folder names and dataset labels lie. This is the "don't trust the metadata" defect, and it reproduced again this session |
| dedup | sha256 exact + 5-word shingles / 64-perm MinHash LSH @ 0.8 | 1,092 + 777 | 0.8 Jaccard catches vendored code copied between repos without merging genuinely distinct files. Near-dupes matter because they silently inflate *unique* supply, which the whole epoch cap rests on |
| PII | email · phone · IPv4 · Aadhaar-like; **code lane exempt from IPv4** | 4,312 redactions | The exemption is deliberate: S4 found the regex layer has real false positives — a UDISE code read as a phone number, and IPv4 patterns are legitimate in source code |
| decontamination | 10-word shingle overlap + BIG-bench canary GUID | 10 GSM8K leaks | **Tuned, not arbitrary:** 5-word shingles produced **79 false positives** in S4. Ten words is the length at which overlap stops being coincidence |
| banding | difficulty as a quantile **within script group**; trace length by reasoning tokens | per document | A fixed readability threshold mislabels entire languages, because a Malayalam word is far longer in code points than an English one |
| reserve | admission criterion per lane → **physically separate shard** + content hash | 6 shards | A flag in a config can be mis-weighted by a sampler. A separate file with its own hash cannot be reached by accident |

**Licensing.** Permissive-only, with a per-file provenance record (url, licence, sha256, byte count,
timestamp) for all 31 sources. Where a licence needs a per-split check rather than a blanket one — xLAM
is the live example — the inventory flags it rather than assuming.

**Evaluation.** Per-lane **decontaminated held-out sets**, plus an **out-of-distribution slice** for
the three lanes where in-distribution scoring would reward format memorisation (agentic → BFCL,
reasoning → PRM800K, code → five unseen repositories). §7 shows why that distinction is not academic:
it flipped a result.

**Monitoring and rollback.** Every stage transition blends over an **8B-token warm-up**. Trigger:
gradient norm rising **>3× over a 2B-token window** ⇒ roll back and re-enter over 16B. V4's incident is
the precedent — a step change in the Hindi share against settled embeddings moved the gradient norm
~150×.

---

## What would change my mind

| risk | trigger | action | why that threshold |
|---|---|---|---|
| **Reasoning supply fails quality gating** | pooled survival **< 65.3%** | share drops to what 4 epochs of *measured* supply funds | 65.3% is not a preference — it is where 222B ÷ (85B × s) crosses 4.0 epochs. Below it the share is arithmetically unfundable |
| **Agentic 2% confirmed at 1B** | already true at 42M, 3 seeds of 3, disjoint ranges | share → 2.0% **with** synthesis already funded to 17B | 2% needs 68B; at 4 epochs that requires 17B of supply. The share and its supply move together or not at all |
| Code-heavy confirmed at 1B | already wins by **3.1×** the noise floor | move 8 points web → code | 3.1× clears the measured 0.0022 resolution; anything under ~2× would be unresolvable noise. Supply allows it — 32% is 0.84 epochs |
| Verified Indic under-delivers | < 70B at D-30 | Tier A 28% → 22%, absorbed by **B**; C and D are **not** raised | 70B is where Tier A's 160.7B demand hits 2.3 epochs. B absorbs it because raising C or D is exactly the translationese failure the split exists to prevent |
| L3/L4 distillation under-delivers | < 15B verified long traces | L3+L4 drops 28% → 18% of the lane | 70.1B of L3/L4 demand ÷ 15B = 4.7 epochs, over the cap. A shallower model beats one trained on unverified long traces |
| Reserve leaks into the main run | any `reserve=true` shard in a main-run manifest | **burn the shard**, do not re-quarantine | Its entire value was in being unseen. Once seen, re-quarantining it is self-deception |

**The sharpest question a reviewer can ask** is the reasoning lane's supply: 85B is a *gross* figure,
our own cleaning measured a 75.5% quality-gate drop against it, and the share needs 65.3% survival to
be fundable. It is priced as a stress test with a sampling gate before stage 1, not defended.

---

## Where the detail lives

| | |
|---|---|
| [`walkthrough.html`](walkthrough.html) | the visual brief — every section of the plan, with the charts |
| [`mixture/v5_mixture.json`](mixture/v5_mixture.json) | shares, tiers, floor, reserve, curriculum, bands, proxy protocol |
| [`mixture/inventory.json`](mixture/inventory.json) | supply, with an evidence tag on every line |
| [`results/mixture_report.md`](results/mixture_report.md) | the solved mixture + the reasoning quality-gate stress test |
| [`results/assignment_audit.md`](results/assignment_audit.md) | all 50 checks, each against a real artifact |
| [`results/proxy_report_gpu.md`](results/proxy_report_gpu.md) · [`_masked`](results/proxy_report_gpu_masked.md) | the GPU runs and the noise floor |
| [`results/selector_report.md`](results/selector_report.md) | retention per lane, and the floor it implies |
| [`results/band_examples.md`](results/band_examples.md) | verbatim text for every difficulty and length band |
| [`results/cleaning_report.md`](results/cleaning_report.md) | per-stage counts, provenance for all 31 sources |

**Sources the numbers lean on:** Muennighoff et al. *Scaling Data-Constrained LMs*
([2305.16264](https://arxiv.org/abs/2305.16264)) — the 4-epoch cap · Liu et al. *RegMix*
([2407.01492](https://arxiv.org/abs/2407.01492)) — proxy mixture search and the web-correlation warning
· Xie et al. *DoReMi* ([2305.10429](https://arxiv.org/abs/2305.10429)) · Khan et al. *IndicLLMSuite*
([2403.06350](https://arxiv.org/abs/2403.06350)) · *FineWeb* ([2406.17557](https://arxiv.org/abs/2406.17557))
· *DCLM* ([2406.11794](https://arxiv.org/abs/2406.11794)) · *StarCoder2*
([2402.19173](https://arxiv.org/abs/2402.19173)) · *ToolLLM* ([2307.16789](https://arxiv.org/abs/2307.16789))
· *xLAM* ([2409.03215](https://arxiv.org/abs/2409.03215)).
