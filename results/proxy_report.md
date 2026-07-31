# Proxy ablation - what actually ran

_6 arms x 11,417,088 params x 2,998,272 tokens, context 256, identical seed and identical token budget; only the mixture differs. CPU, 72.5 minutes total._

**W** = capability-weighted held-out NLL, each lane normalised to the best arm (lower is better): 0.30*code + 0.20*agentic + 0.20*indic + 0.15*reasoning + 0.10*long_context + 0.05*general_web.

| arm | W | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `indic_heavy` | **1.0246** | 5.513 | 6.284 | 4.080 | 6.206 | 4.618 | 6.244 | 5.298 |
| `code_heavy` | **1.0249** | 5.295 | 6.304 | 4.294 | 6.218 | 4.647 | 6.230 | 5.332 |
| `s5_composer_default` | **1.0300** | 5.474 | 6.241 | 4.349 | 6.225 | 4.760 | 6.227 | 5.059 |
| `v5_proposed` | **1.0379** | 5.476 | 6.250 | 4.296 | 6.232 | 4.671 | 6.206 | 5.399 |
| `indic_lite` | **1.0633** | 5.513 | 6.204 | 4.636 | 6.255 | 4.762 | 6.213 | 5.501 |
| `naive_web_heavy` | **1.1981** | 6.067 | 6.030 | 4.772 | 6.591 | 5.431 | 6.298 | 7.397 |

(cells are held-out cross-entropy in nats/token on that lane's decontaminated hold-out, which no arm trained on)

> Do not compare cells *across* lanes. A lane's absolute loss is confounded by its tokenizer fertility and by how templated its sources are - Indic Wikipedia is cheaper per token than English Wikipedia here largely because our 32k BPE spends 3.6-7.5 tokens on an Indic word and 1.8 on an English one. `W` normalises each lane to the best arm on that same lane, so the confound cancels; the raw column does not.

## Spread: does the mixture matter at all?

| lane | best arm | worst arm | spread (nats) | spread (%) |
|---|---|---|---:|---:|
| code | code_heavy (5.295) | naive_web_heavy (6.067) | 0.773 | 14.6% |
| general_web | naive_web_heavy (6.030) | code_heavy (6.304) | 0.275 | 4.6% |
| indic | indic_heavy (4.080) | naive_web_heavy (4.772) | 0.693 | 17.0% |
| stem_math | indic_heavy (6.206) | naive_web_heavy (6.591) | 0.385 | 6.2% |
| reasoning | indic_heavy (4.618) | naive_web_heavy (5.431) | 0.813 | 17.6% |
| long_context | v5_proposed (6.206) | naive_web_heavy (6.298) | 0.092 | 1.5% |
| agentic | s5_composer_default (5.059) | naive_web_heavy (7.397) | 2.338 | 46.2% |

## Own-share elasticity: what each lane pays for being squeezed

`nll_lane = a + b·ln(share_lane)`, fitted across the arms that gave the lane a non-zero share. **b is nats of held-out loss per e-fold of share** — more negative means the lane is on a steep part of the curve and is expensive to cut.

| lane | b (nats per e-fold) | R² | arms | share range tested |
|---|---:|---:|---:|---|
| reasoning | **-0.604** | 0.98 | 6 | 2.0% – 7.0% |
| code | **-0.543** | 0.98 | 6 | 8.0% – 32.0% |
| agentic | **-0.467** | 0.78 | 5 | 1.0% – 2.0% |
| indic | **-0.429** | 1.00 | 6 | 6.0% – 30.0% |
| stem_math | **-0.341** | 0.99 | 6 | 4.0% – 12.0% |
| general_web | **-0.214** | 0.97 | 6 | 23.0% – 78.0% |
| long_context | **-0.056** | 0.82 | 6 | 2.0% – 7.6% |

> The long-context row measures the instrument, not the lane. At context 256 a model cannot express long-context capability at all, so its loss barely responds to share. Read it as *this proxy is blind to this lane* - which is worth knowing before spending 1B-scale compute on the same design: the stage-1 arms have to run at a context long enough for the lane to mean anything.

## Is the mixture at a local optimum?

Moving **3 points of share** from one lane to another and predicting the change in the capability-weighted objective `W` from the fitted curves. Negative = the transfer would improve the plan.

| transfer | ΔW |
|---|---:|
| stem_math → agentic | **-0.0256** |
| general_web → agentic | **-0.0254** |
| long_context → agentic | **-0.0251** |
| indic → agentic | **-0.0217** |
| code → agentic | **-0.0215** |
| … | |
| reasoning → long_context | +0.0107 |
| reasoning → general_web | +0.0108 |
| reasoning → stem_math | +0.0110 |

> **The proxy does not endorse the mixture as written.** Its best single move is 3 points from `stem_math` to `agentic` (ΔW = -0.0256). At this scale that is a direction, not a decision — the fit has 6 points, the models are 11.4M parameters, and the agentic slope is fitted over a 1–2% range in which repetition is free (§8.3 removes that) — but it is exactly the hypothesis the 1B arm list in §8.1 has to settle, and it is written down here before the 1B runs rather than after.

## What this proxy is not

Three orders of magnitude below the pre-registered 1B protocol, on 11.4M-parameter models trained for 3.0M tokens each. It cannot rank arms on downstream benchmarks — nothing at this size produces a non-trivial HumanEval or MILU score — and RegMix's own caution applies doubly here: domain interactions are complicated and small-model orderings can invert. It is reported because it is real, it was cheap, it is reproducible from this repo, and it puts numbers where the plan would otherwise have adjectives.
