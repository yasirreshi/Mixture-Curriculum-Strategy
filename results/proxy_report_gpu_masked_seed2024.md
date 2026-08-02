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
| 1 | `v5_agentic_2pct` | **1.0006** | [1.0001, 1.0014] | 1.0006 | 5.833 | 5.247 | 4.211 | 5.495 | 4.128 | 5.924 | 4.497 |
| 2 | `v5_proposed` | **1.0087** | [1.0072, 1.0103] | 1.0083 | 5.827 | 5.239 | 4.272 | 5.503 | 4.121 | 5.928 | 4.625 |

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
| agentic | -0.1283 | [-0.1601, -0.0972] | 1.000 |
| reasoning | +0.0072 | [-0.0012, +0.0157] | 0.046 |
| code | +0.0052 | [-0.0071, +0.0173] | 0.201 |
| indic | -0.0603 | [-0.0791, -0.0436] | 1.000 |

W: `v5_agentic_2pct` 1.0007 [1.0001, 1.0014] vs `v5_proposed` 1.0087 [1.0072, 1.0103] - CIs **do not overlap**.

## What this instrument can actually resolve

`v5_proposed` was retrained at 3 different seeds with the **identical** mixture. Everything that separates those runs is noise, so the spread across them is the smallest difference between two arms that can be believed. A confidence interval over held-out sequences does not capture this - it resamples the test set, not the training run.

| lane | spread across seeds (nats) |
|---|---:|
| agentic | 0.2813 |
| reasoning | 0.0583 |
| indic | 0.0548 |
| code | 0.0455 |
| stem_math | 0.0266 |
| general_web | 0.0244 |
| long_context | 0.0118 |

**Noise floor on `W`: 0.0098** (1.0114 – 1.0211 across seeds).

| arm | gap to `v5_proposed` on W | multiples of the noise floor | verdict |
|---|---:|---:|---|
| `v5_agentic_2pct` | 0.0080 | 0.8x | **below the noise floor** |

## Per-lane spread

| lane | best arm | worst arm | spread (%) |
|---|---|---|---:|
| code | v5_proposed (5.827) | v5_agentic_2pct (5.833) | 0.1% |
| general_web | v5_proposed (5.239) | v5_agentic_2pct (5.247) | 0.2% |
| indic | v5_agentic_2pct (4.211) | v5_proposed (4.272) | 1.4% |
| stem_math | v5_agentic_2pct (5.495) | v5_proposed (5.503) | 0.1% |
| reasoning | v5_proposed (4.121) | v5_agentic_2pct (4.128) | 0.2% |
| long_context | v5_agentic_2pct (5.924) | v5_proposed (5.928) | 0.1% |
| agentic | v5_agentic_2pct (4.497) | v5_proposed (4.625) | 2.9% |
