# Selector bias, measured

_Base model 11,482,624 params trained on a balanced 7-lane mixture; proxy direction built from general_web 45%, code 40%, stem_math 15%; 90 candidate batches per lane; top 40% retained (V4's retained fraction)._

## scoring on the **prefix 500**

| lane | mean cosine to proxy | retained @top-40% | scheduled share | realized share | delta |
|---|---:|---:|---:|---:|---:|
| code | +0.1437 | 62.2% | 24.0% | 29.0% | +5.0pt |
| general_web | +0.1658 | 85.6% | 31.0% | 51.6% | +20.6pt |
| indic | -0.0054 | 0.0% | 17.9% | 0.0% | -17.9pt |
| stem_math | +0.1134 | 52.2% | 11.5% | 11.7% | +0.2pt |
| reasoning | +0.0754 | 23.3% | 7.0% | 3.2% | -3.8pt |
| long_context | +0.0743 | 26.7% | 7.6% | 3.9% | -3.7pt |
| agentic | +0.0890 | 30.0% | 1.0% | 0.6% | -0.4pt |

Floor needed to hold realized share within 1pt of scheduled: **indic 16.90%**, **agentic 0.00%**, **reasoning 5.70%**

## scoring on the **full document window**

| lane | mean cosine to proxy | retained @top-40% | scheduled share | realized share | delta |
|---|---:|---:|---:|---:|---:|
| code | +0.1784 | 71.1% | 24.0% | 33.5% | +9.5pt |
| general_web | +0.1638 | 74.4% | 31.0% | 45.3% | +14.3pt |
| indic | +0.0116 | 1.1% | 17.9% | 0.4% | -17.5pt |
| stem_math | +0.1084 | 28.9% | 11.5% | 6.5% | -5.0pt |
| reasoning | +0.0977 | 15.6% | 7.0% | 2.1% | -4.9pt |
| long_context | +0.1621 | 81.1% | 7.6% | 12.1% | +4.5pt |
| agentic | +0.0628 | 7.8% | 1.0% | 0.2% | -0.8pt |

Floor needed to hold realized share within 1pt of scheduled: **indic 16.89%**, **agentic 0.00%**, **reasoning 5.82%**
