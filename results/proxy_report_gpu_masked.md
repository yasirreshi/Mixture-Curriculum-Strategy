# Proxy ablation on GPU - the pre-registered corrections, run

_3 arms x 42,128,384 params x 27,992,064 tokens, context 1024, identical seed and identical budget; only the mixture differs. 10.4 GPU-minutes total on NVIDIA RTX 5000 Ada Generation Laptop GPU._

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
| 1 | `code_heavy` | **1.0034** | [1.0024, 1.0044] | 1.0034 | 5.662 | 5.315 | 4.211 | 5.495 | 4.110 | 5.961 | 4.540 |
| 2 | `v5_agentic_2pct` | **1.0075** | [1.0067, 1.0084] | 1.0073 | 5.774 | 5.243 | 4.209 | 5.497 | 4.130 | 5.940 | 4.508 |
| 3 | `v5_proposed` | **1.0094** | [1.0082, 1.0106] | 1.0089 | 5.754 | 5.232 | 4.201 | 5.479 | 4.134 | 5.920 | 4.591 |

`W` is the pre-registered objective (stem_math weight 0). `W_stem` gives stem_math weight 0.05 with the others renormalised - reported because an objective blind to an 11.5% lane cannot price transfers out of it.

## Epochs actually spent (the caps biting)

| arm | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|
| `code_heavy` | 0.84 | 0.15 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |
| `v5_agentic_2pct` | 0.63 | 0.19 | 1.88 | 1.33 | 2.39 | 2.20 | 5.46 |
| `v5_proposed` | 0.63 | 0.20 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |

## The pre-registered agentic decision

§8.3 committed to a rule before this run: *if the run with an OOD agentic hold-out still favours 2% over 1%, the agentic share rises to 2.0% and the synthesis target rises 10B -> 17B.* Here is the paired test.

| lane | v5_agentic_2pct - v5_proposed (nats) | 95% CI | P(2% better) |
|---|---:|---|---:|
| agentic | -0.0825 | [-0.1057, -0.0582] | 1.000 |
| reasoning | -0.0042 | [-0.0129, +0.0051] | 0.817 |
| code | +0.0204 | [+0.0078, +0.0330] | 0.001 |
| indic | +0.0085 | [+0.0010, +0.0158] | 0.015 |

W: `v5_agentic_2pct` 1.0075 [1.0067, 1.0084] vs `v5_proposed` 1.0094 [1.0082, 1.0106] - CIs **overlap**.

## What this instrument can actually resolve

`v5_agentic_2pct` was retrained at 3 different seeds with the **identical** mixture. Everything that separates those runs is noise, so the spread across them is the smallest difference between two arms that can be believed. A confidence interval over held-out sequences does not capture this - it resamples the test set, not the training run.

| lane | spread across seeds (nats) |
|---|---:|
| code | 0.0632 |
| agentic | 0.0453 |
| long_context | 0.0271 |
| general_web | 0.0239 |
| stem_math | 0.0130 |
| reasoning | 0.0116 |
| indic | 0.0027 |

**Noise floor on `W`: 0.0058** (1.0064 – 1.0122 across seeds).

| arm | gap to `v5_agentic_2pct` on W | multiples of the noise floor | verdict |
|---|---:|---:|---|
| `v5_proposed` | 0.0019 | 0.3x | **below the noise floor** |
| `code_heavy` | 0.0041 | 0.7x | **below the noise floor** |

## Per-lane spread

| lane | best arm | worst arm | spread (%) |
|---|---|---|---:|
| code | code_heavy (5.662) | v5_agentic_2pct (5.774) | 2.0% |
| general_web | v5_proposed (5.232) | code_heavy (5.315) | 1.6% |
| indic | v5_proposed (4.201) | code_heavy (4.211) | 0.2% |
| stem_math | v5_proposed (5.479) | v5_agentic_2pct (5.497) | 0.3% |
| reasoning | code_heavy (4.110) | v5_proposed (4.134) | 0.6% |
| long_context | v5_proposed (5.920) | code_heavy (5.961) | 0.7% |
| agentic | v5_agentic_2pct (4.508) | v5_proposed (4.591) | 1.8% |
