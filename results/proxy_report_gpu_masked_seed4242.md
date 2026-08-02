# Proxy ablation on GPU - the pre-registered corrections, run

_2 arms x 42,128,384 params x 27,992,064 tokens, context 1024, identical seed and identical budget; only the mixture differs. 7.0 GPU-minutes total on NVIDIA RTX 5000 Ada Generation Laptop GPU._

Loss masking (§4.1 convention): **on**. Per-lane unique-supply caps: **on** (epoch-honest).

## What each lane's hold-out actually is

| lane | hold-out | sequences |
|---|---|---:|
| code | OOD (BurntSushi/ripgrep, expressjs/express, redis/redis, sharkdp/fd, spf13/cobra) | 160 |
| general_web | in-distribution (held-out documents) | 160 |
| indic | in-distribution (held-out documents) | 160 |
| stem_math | in-distribution (held-out documents) | 160 |
| reasoning | OOD (prm800k-phase2) | 160 |
| long_context | in-distribution (held-out documents) | 160 |
| agentic | OOD (bfcl-v4-live-multiple, bfcl-v4-multi-turn, bfcl-v4-parallel) | 160 |

> The three lanes with an OOD hold-out are the three the plan said it could not adjudicate without one. For those, no training token comes from the source family the arm is scored on, so an arm cannot win by memorising a format. The other four are in-distribution document splits and are labelled as such - no OOD claim is made for them.

## Result

| rank | arm | W | 95% CI | W_stem | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `v5_agentic_2pct` | **1.0004** | [1.0000, 1.0008] | 1.0004 | 5.769 | 5.223 | 4.209 | 5.484 | 4.118 | 5.913 | 4.463 |
| 2 | `v5_proposed` | **1.0055** | [1.0042, 1.0067] | 1.0052 | 5.811 | 5.233 | 4.215 | 5.483 | 4.109 | 5.910 | 4.527 |

`W` is the pre-registered objective (stem_math weight 0). `W_stem` gives stem_math weight 0.05 with the others renormalised - reported because an objective blind to an 11.5% lane cannot price transfers out of it.

## Epochs actually spent (the caps biting)

| arm | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v5_agentic_2pct` | 0.63 | 0.19 | 1.88 | 1.33 | 2.39 | 2.20 | 5.46 |
| `v5_proposed` | 0.63 | 0.20 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |

## The pre-registered agentic decision

§8.3 committed to a rule before this run: *if the run with an OOD agentic hold-out still favours 2% over 1%, the agentic share rises to 2.0% and the synthesis target rises 10B -> 17B.* Here is the paired test.

| lane | v5_agentic_2pct - v5_proposed (nats) | 95% CI | P(2% better) |
|---|---:|---|---:|
| agentic | -0.0643 | [-0.0867, -0.0419] | 1.000 |
| reasoning | +0.0088 | [-0.0027, +0.0198] | 0.069 |
| code | -0.0421 | [-0.0538, -0.0298] | 1.000 |
| indic | -0.0063 | [-0.0141, +0.0014] | 0.950 |

W: `v5_agentic_2pct` 1.0004 [1.0000, 1.0008] vs `v5_proposed` 1.0055 [1.0042, 1.0067] - CIs **do not overlap**.

## What this instrument can actually resolve

`v5_proposed` was retrained at 3 different seeds with the **identical** mixture. Everything that separates those runs is noise, so the spread across them is the smallest difference between two arms that can be believed. A confidence interval over held-out sequences does not capture this - it resamples the test set, not the training run.

| lane | spread across seeds (nats) |
|---|---:|
| agentic | 0.3793 |
| reasoning | 0.0699 |
| indic | 0.0345 |
| code | 0.0296 |
| long_context | 0.0190 |
| general_web | 0.0185 |
| stem_math | 0.0064 |

**Noise floor on `W`: 0.0186** (1.0055 – 1.0241 across seeds).

| arm | gap to `v5_proposed` on W | multiples of the noise floor | verdict |
|---|---:|---:|---|
| `v5_agentic_2pct` | 0.0051 | 0.3x | **below the noise floor** |

## Per-lane spread

| lane | best arm | worst arm | spread (%) |
|---|---|---|---:|
| code | v5_agentic_2pct (5.769) | v5_proposed (5.811) | 0.7% |
| general_web | v5_agentic_2pct (5.223) | v5_proposed (5.233) | 0.2% |
| indic | v5_agentic_2pct (4.209) | v5_proposed (4.215) | 0.1% |
| stem_math | v5_proposed (5.483) | v5_agentic_2pct (5.484) | 0.0% |
| reasoning | v5_proposed (4.109) | v5_agentic_2pct (4.118) | 0.2% |
| long_context | v5_proposed (5.910) | v5_agentic_2pct (5.913) | 0.1% |
| agentic | v5_agentic_2pct (4.463) | v5_proposed (4.527) | 1.4% |
