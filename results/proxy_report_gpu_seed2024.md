# Proxy ablation on GPU - the pre-registered corrections, run

_1 arms x 42,128,384 params x 27,992,064 tokens, context 1024, identical seed and identical budget; only the mixture differs. 3.5 GPU-minutes total on NVIDIA RTX 5000 Ada Generation Laptop GPU._

Loss masking (§4.1 convention): **off**. Per-lane unique-supply caps: **on** (epoch-honest).

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
| 1 | `v5_proposed` | **1.0000** | [1.0000, 1.0000] | 1.0000 | 5.782 | 5.214 | 4.249 | 5.476 | 4.145 | 5.928 | 4.907 |

`W` is the pre-registered objective (stem_math weight 0). `W_stem` gives stem_math weight 0.05 with the others renormalised - reported because an objective blind to an 11.5% lane cannot price transfers out of it.

## Epochs actually spent (the caps biting)

| arm | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v5_proposed` | 0.63 | 0.20 | 1.88 | 1.33 | 2.39 | 2.20 | 2.73 |

## Per-lane spread

| lane | best arm | worst arm | spread (%) |
|---|---|---|---:|
| code | v5_proposed (5.782) | v5_proposed (5.782) | 0.0% |
| general_web | v5_proposed (5.214) | v5_proposed (5.214) | 0.0% |
| indic | v5_proposed (4.249) | v5_proposed (4.249) | 0.0% |
| stem_math | v5_proposed (5.476) | v5_proposed (5.476) | 0.0% |
| reasoning | v5_proposed (4.145) | v5_proposed (4.145) | 0.0% |
| long_context | v5_proposed (5.928) | v5_proposed (5.928) | 0.0% |
| agentic | v5_proposed (4.907) | v5_proposed (4.907) | 0.0% |
