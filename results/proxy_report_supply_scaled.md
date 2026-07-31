# Proxy ablation - what actually ran

_4 arms x 11,417,088 params x 2,998,272 tokens, context 256, identical seed and identical token budget; only the mixture differs. CPU, 63.1 minutes total._

**W** = capability-weighted held-out NLL, each lane normalised to the best arm (lower is better): 0.30*code + 0.20*agentic + 0.20*indic + 0.15*reasoning + 0.10*long_context + 0.05*general_web.

| arm | W | code | general_web | indic | stem_math | reasoning | long_context | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_agentic_2pct` | **1.0165** | 5.580 | 6.299 | 4.335 | 6.273 | 4.818 | 6.241 | 5.193 |
| `code_heavy` | **1.0226** | 5.461 | 6.385 | 4.359 | 6.289 | 4.771 | 6.285 | 5.492 |
| `indic_heavy` | **1.0236** | 5.627 | 6.360 | 4.177 | 6.287 | 4.767 | 6.274 | 5.521 |
| `v5_proposed` | **1.0243** | 5.563 | 6.273 | 4.316 | 6.266 | 4.759 | 6.221 | 5.505 |

(cells are held-out cross-entropy in nats/token on that lane's decontaminated hold-out, which no arm trained on)

> Do not compare cells *across* lanes. A lane's absolute loss is confounded by its tokenizer fertility and by how templated its sources are - Indic Wikipedia is cheaper per token than English Wikipedia here largely because our 32k BPE spends 3.6-7.5 tokens on an Indic word and 1.8 on an English one. `W` normalises each lane to the best arm on that same lane, so the confound cancels; the raw column does not.

## Spread: does the mixture matter at all?

| lane | best arm | worst arm | spread (nats) | spread (%) |
|---|---|---|---:|---:|
| code | code_heavy (5.461) | indic_heavy (5.627) | 0.166 | 3.0% |
| general_web | v5_proposed (6.273) | code_heavy (6.385) | 0.112 | 1.8% |
| indic | indic_heavy (4.177) | code_heavy (4.359) | 0.182 | 4.4% |
| stem_math | v5_proposed (6.266) | code_heavy (6.289) | 0.023 | 0.4% |
| reasoning | v5_proposed (4.759) | v5_agentic_2pct (4.818) | 0.059 | 1.3% |
| long_context | v5_proposed (6.221) | code_heavy (6.285) | 0.065 | 1.0% |
| agentic | v5_agentic_2pct (5.193) | indic_heavy (5.521) | 0.328 | 6.3% |
