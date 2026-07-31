# Selector bias, measured

_Base model 11,482,624 params trained on a balanced 7-lane mixture; proxy direction built from general_web 45%, code 40%, stem_math 15%; 90 candidate batches per lane; top 40% retained (V4's retained fraction)._

## scoring on the **prefix 500**

| lane | mean cosine to proxy | retained @top-40% | scheduled share | realized share | delta |
|---|---:|---:|---:|---:|---:|
| code | +0.1251 | 68.9% | 24.0% | 29.3% | +5.3pt |
| general_web | +0.1779 | 93.3% | 31.0% | 51.2% | +20.2pt |
| indic | +0.0062 | 0.0% | 17.9% | 0.0% | -17.9pt |
| stem_math | +0.1248 | 81.1% | 11.5% | 16.5% | +5.0pt |
| reasoning | +0.0726 | 22.2% | 7.0% | 2.8% | -4.2pt |
| long_context | +0.0050 | 0.0% | 7.6% | 0.0% | -7.6pt |
| agentic | +0.0586 | 14.4% | 1.0% | 0.3% | -0.7pt |

Floor needed to hold realized share within 1pt of scheduled: **indic 16.90%**, **agentic 0.00%**, **reasoning 5.71%**

## scoring on the **full document window**

| lane | mean cosine to proxy | retained @top-40% | scheduled share | realized share | delta |
|---|---:|---:|---:|---:|---:|
| code | +0.1720 | 64.4% | 24.0% | 31.0% | +7.0pt |
| general_web | +0.1638 | 70.0% | 31.0% | 43.5% | +12.5pt |
| indic | +0.0084 | 1.1% | 17.9% | 0.4% | -17.5pt |
| stem_math | +0.1264 | 47.8% | 11.5% | 11.0% | -0.5pt |
| reasoning | +0.0840 | 15.6% | 7.0% | 2.2% | -4.8pt |
| long_context | +0.1687 | 77.8% | 7.6% | 11.8% | +4.2pt |
| agentic | +0.0551 | 3.3% | 1.0% | 0.1% | -0.9pt |

Floor needed to hold realized share within 1pt of scheduled: **indic 16.89%**, **agentic 0.00%**, **reasoning 5.82%**
