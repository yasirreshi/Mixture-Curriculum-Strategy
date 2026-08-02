# Proxy ablation on GPU - the pre-registered corrections, run

_7 arms x 42,128,384 params x 27,992,064 tokens, context 1024, identical seed and identical budget; only the mixture differs. 37.9 GPU-minutes total on NVIDIA RTX 5000 Ada Generation Laptop GPU._

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
| 1 | `code_heavy` | **1.0136** | [1.0128, 1.0145] | 1.0131 | 5.625 | 5.294 | 4.195 | 5.467 | 4.164 | 5.970 | 4.900 |
| 2 | `v5_agentic_2pct` | **1.0196** | [1.0185, 1.0207] | 1.0187 | 5.758 | 5.217 | 4.191 | 5.462 | 4.157 | 5.911 | 4.928 |
| 3 | `v5_proposed` | **1.0205** | [1.0195, 1.0215] | 1.0197 | 5.748 | 5.221 | 4.196 | 5.482 | 4.189 | 5.918 | 4.925 |
| 4 | `s5_composer_default` | **1.0242** | [1.0227, 1.0256] | 1.0230 | 5.753 | 5.209 | 4.279 | 5.452 | 4.209 | 5.933 | 4.887 |
| 5 | `indic_heavy` | **1.0287** | [1.0278, 1.0297] | 1.0278 | 5.843 | 5.333 | 4.075 | 5.514 | 4.198 | 5.984 | 5.082 |
| 6 | `indic_lite` | **1.0428** | [1.0410, 1.0446] | 1.0409 | 5.778 | 5.159 | 4.529 | 5.472 | 4.192 | 5.915 | 5.045 |
| 7 | `naive_web_heavy` | **1.0974** | [1.0941, 1.1006] | 1.0942 | 6.206 | 4.810 | 4.627 | 5.634 | 4.547 | 5.856 | 5.504 |

`W` is the pre-registered objective (stem_math weight 0). `W_stem` gives stem_math weight 0.05 with the others renormalised - reported because an objective blind to an 11.5% lane cannot price transfers out of it.

## Epochs actually spent (the caps biting)

| arm | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|
| `code_heavy` | 0.84 | 0.15 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |
| `v5_agentic_2pct` | 0.63 | 0.19 | 1.88 | 1.33 | 2.39 | 2.20 | 5.46 |
| `v5_proposed` | 0.63 | 0.20 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |
| `s5_composer_default` | 0.63 | 0.22 | 1.68 | 1.39 | 2.05 | 1.74 | 5.46 |
| `indic_heavy` | 0.53 | 0.15 | 3.15 | 1.33 | 2.39 | 2.20 | 2.73 |
| `indic_lite` | 0.63 | 0.26 | 0.84 | 1.33 | 2.39 | 2.20 | 2.73 |
| `naive_web_heavy` | 0.21 | 0.50 | 0.63 | 0.46 | 0.68 | 0.58 | 0.00 |

## The pre-registered agentic decision

§8.3 committed to a rule before this run: *if the run with an OOD agentic hold-out still favours 2% over 1%, the agentic share rises to 2.0% and the synthesis target rises 10B -> 17B.* Here is the paired test.

| lane | v5_agentic_2pct - v5_proposed (nats) | 95% CI | P(2% better) |
|---|---:|---|---:|
| agentic | +0.0028 | [-0.0059, +0.0112] | 0.253 |
| reasoning | -0.0314 | [-0.0395, -0.0232] | 1.000 |
| code | +0.0105 | [-0.0007, +0.0220] | 0.034 |
| indic | -0.0057 | [-0.0125, +0.0011] | 0.951 |

W: `v5_agentic_2pct` 1.0196 [1.0185, 1.0207] vs `v5_proposed` 1.0205 [1.0195, 1.0215] - CIs **overlap**.

## The pre-registered kill condition

*Refutes if the naive web-heavy arm lands within CI of V5-proposed on W.* `naive_web_heavy` 1.0974 [1.0941, 1.1006] vs `v5_proposed` 1.0205 [1.0195, 1.0215] -> **did not fire**.

## What this instrument can actually resolve

`v5_proposed` was retrained at 3 different seeds with the **identical** mixture. Everything that separates those runs is noise, so the spread across them is the smallest difference between two arms that can be believed. A confidence interval over held-out sequences does not capture this - it resamples the test set, not the training run.

| lane | spread across seeds (nats) |
|---|---:|
| indic | 0.0530 |
| agentic | 0.0491 |
| code | 0.0444 |
| reasoning | 0.0435 |
| general_web | 0.0180 |
| long_context | 0.0118 |
| stem_math | 0.0053 |

**Noise floor on `W`: 0.0022** (1.0214 – 1.0236 across seeds).

| arm | gap to `v5_proposed` on W | multiples of the noise floor | verdict |
|---|---:|---:|---|
| `v5_agentic_2pct` | 0.0009 | 0.4x | **below the noise floor** |
| `s5_composer_default` | 0.0037 | 1.7x | **below the noise floor** |
| `code_heavy` | 0.0069 | 3.1x | resolvable |
| `indic_heavy` | 0.0082 | 3.7x | resolvable |
| `indic_lite` | 0.0223 | 10.1x | resolvable |
| `naive_web_heavy` | 0.0770 | 34.8x | resolvable |

## Per-lane spread

| lane | best arm | worst arm | spread (%) |
|---|---|---|---:|
| code | code_heavy (5.625) | naive_web_heavy (6.206) | 10.3% |
| general_web | naive_web_heavy (4.810) | indic_heavy (5.333) | 10.9% |
| indic | indic_heavy (4.075) | naive_web_heavy (4.627) | 13.5% |
| stem_math | s5_composer_default (5.452) | naive_web_heavy (5.634) | 3.3% |
| reasoning | v5_agentic_2pct (4.157) | naive_web_heavy (4.547) | 9.4% |
| long_context | naive_web_heavy (5.856) | indic_heavy (5.984) | 2.2% |
| agentic | s5_composer_default (4.887) | naive_web_heavy (5.504) | 12.6% |
