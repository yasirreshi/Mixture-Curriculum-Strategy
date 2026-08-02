# Difficulty and reasoning-length bands, with real examples

_Every example below is a document from `data/clean/` in this repo, with the token count measured by the 32k BPE trained in `scripts/02_clean_lanes.py`. Band assignment is done by the documented heuristics in that script, not by hand._

## Difficulty ladder

| band | name | docs in corpus | lanes present | median tokens |
|---|---|---:|---|---:|
| **B0** | Nursery | 854 | general_web, indic | 3,448 |
| **B1** | Grade-school | 16,966 | code, general_web, indic, reasoning | 180 |
| **B2** | High-school | 29,065 | agentic, code, general_web, indic, reasoning | 343 |
| **B3** | Undergraduate | 6,076 | agentic, code, general_web, indic, long_context, reasoning | 673 |
| **B4** | Graduate | 3,929 | agentic, code, general_web, indic, long_context, reasoning | 4,238 |
| **B5** | Research / PhD | 24,100 | agentic, general_web, indic, reasoning, stem_math | 224 |

### B0 - Nursery

- **lane** `general_web` · **source** `wikipedia:en` · **3,214 tokens** (3,214 supervised / 0 context) · lang `en`
  > The Republic of Estonia gained its independence from the Russian Empire on 24 February 1918 and established diplomatic relations with many countries via membership of the League of Nations. The forcible incorporation of Estonia into the Soviet Union in 1940 was not generally recognised by the international community and the Estonian diplomatic service continued to operate in some countries. Following the restoration  ...

- **lane** `indic` · **source** `wikipedia:gu` · **1,183 tokens** (1,183 supervised / 0 context) · lang `gu`
  > રાજ્ય સભા એ ભારત ના સંસદ નું ઊપલું સદન છે. ભારતીય બંધારણ પ્રમાણે રાજ્ય સભાના ૨૫૦ સભ્યો છે જેમાના ૧૨ સભ્યોની નિમણુંક રાષ્ટ્રપતિ કરે છે. આ સભ્યોને નામાંકિત સભ્યો કહેવાય છે. આ નિમણુંક વિવિધ ક્ષેત્રોના નિષ્ણાતો જેમકે - કલા, સાહિત્ય, વિજ્ઞાન અને સમાજસેવકોમાંથી કરાય છે. બાકીના સભ્યો ભારતનાં રાજ્યો અને કેન્દ્રશાસિત પ્રદેશોના ચૂંટાયેલા સભ્યો કરે છે. રાજ્ય સભાનું ક્યારેય વિસર્જન થતું નથી અને તેના એક તૃતિયાંશ સભ્યો દર ૨ વર્ષે  ...

### B1 - Grade-school

- **lane** `reasoning` · **source** `gsm8k-socratic` · **233 tokens** (173 supervised / 60 context) · lang `en`
  > Problem: Claudia has 122 ounces of water and is filling up cups. She has 8-ounce glasses, 5-ounce glasses, and 4-ounce glasses. If she fills six 5 ounce glasses and four 8 ounce glasses, how many 4 ounce glasses can she fill with the remaining water?
<reasoning>
How many ounces did Claudia use on 8-ounce glasses? ** She has used 32 ounces on 8-ounce glasses because 4 x 8 = 32
How many ounces did Claudia use on 5-ounc ...

- **lane** `code` · **source** `pallets/flask` · **218 tokens** (218 supervised / 0 context) · lang `code`
  > from __future__ import annotations
from blinker import Namespace
# This namespace is only for signals provided by Flask itself.
_signals = Namespace()
template_rendered = _signals.signal("template-rendered")
before_render_template = _signals.signal("before-render-template")
request_started = _signals.signal("request-started")
request_finished = _signals.signal("request-finished")
request_tearing_down = _signals.signa ...

### B2 - High-school

- **lane** `agentic` · **source** `gorilla-apibench-huggingface` · **266 tokens** (236 supervised / 30 context) · lang `en`
  > <user>
There's a huge article on recent advancements in cancer research, and I need a concise summary of it.
</user>
<assistant>
<<<domain>>>: Natural Language Processing Summarization
<<<api_call>>>: pipeline('summarization', model='google/pegasus-pubmed')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. Import the 'pipeline' function from the transformers library, provided by Hugging Face.
2. Cre ...

- **lane** `reasoning` · **source** `prm800k-phase2` · **115 tokens** (88 supervised / 27 context) · lang `en`
  > Problem: Find the average value of $0$, $2z$, $4z$, $8z$, and $16z$.
<reasoning>
I need to find the sum of these numbers and divide by the number of numbers.
The sum is $0 + 2z + 4z + 8z + 16z = 30z$.
There are five numbers, so I divide by $5$.
The average value is $\frac{30z}{5} = 6z$.
# Answer
6z
</reasoning>
Answer: 6z

### B3 - Undergraduate

- **lane** `general_web` · **source** `wikipedia:en` · **10,336 tokens** (10,336 supervised / 0 context) · lang `en`
  > Charles Mingus Jr. (April 22, 1922 – January 5, 1979) was an American jazz upright bassist, composer, bandleader, pianist, and author. A major proponent of collective improvisation, he is considered one of the greatest jazz musicians and composers in history, with a career spanning three decades and collaborations with other jazz greats such as Duke Ellington, Charlie Parker, Max Roach, and Eric Dolphy. Mingus's work ...

- **lane** `agentic` · **source** `gorilla-openfunctions-v1` · **396 tokens** (72 supervised / 324 context) · lang `en`
  > <tools>
{'name': 'RapidAPI', 'api_name': 'requests.get', 'description': 'DigiXpress is a solution for shipping and delivering parcels and letters through a vast network of relay points and door-to-door; based on a parcel rating algorithm knowing only the name of the parcel.', 'parameters': []}
{'name': 'pyarrow', 'api_name': 'compare_ordered_inequality', 'description': 'Compare values for ordered inequality', 'parame ...

### B4 - Graduate

- **lane** `code` · **source** `numpy/numpy` · **10,302 tokens** (10,302 supervised / 0 context) · lang `code`
  > import warnings
import pytest
import numpy as np
from numpy import histogram, histogram_bin_edges, histogramdd
from numpy.testing import (
 assert_,
 assert_allclose,
 assert_almost_equal,
 assert_array_almost_equal,
 assert_array_equal,
 assert_array_max_ulp,
 assert_equal,
 assert_raises,
 assert_raises_regex,
)
class TestHistogram:
 def setup_method(self):
 pass
 def teardown_method(self):
 pass
 def test_simple(s ...

- **lane** `general_web` · **source** `wikipedia:en` · **959 tokens** (959 supervised / 0 context) · lang `en`
  > In mathematics, a binary relation R {\displaystyle R} on a set X {\displaystyle X} is antisymmetric if there is no pair of distinct elements of X {\displaystyle X} each of which is related by R {\displaystyle R} to the other. More formally, R {\displaystyle R} is antisymmetric precisely if for all a , b ∈ X , {\displaystyle a,b\in X,} if  a R b  with  a ≠ b  then  b R a  must not hold , {\displaystyle {\text{if }}\,a ...

### B5 - Research / PhD

- **lane** `stem_math` · **source** `arxiv` · **312 tokens** (312 supervised / 0 context) · lang `en`
  > Randomness-efficient Curve Samplers Curve samplers are sampling algorithms that proceed by viewing the domain as a vector space over a finite field, and randomly picking a low-degree curve in it as the sample. Curve samplers exhibit a nice property besides the sampling property: the restriction of low-degree polynomials over the domain to the sampled curve is still low-degree. This property is often used in combinati ...

- **lane** `general_web` · **source** `wikipedia:en` · **10,812 tokens** (10,812 supervised / 0 context) · lang `en`
  > August 28 is the 240th day of the year (241st in leap years) in the Gregorian calendar; 125 days remain until the end of the year. 475 – The Roman general Orestes forces western Roman Emperor Julius Nepos to flee his capital city, Ravenna. 489 – Theodoric, king of the Ostrogoths, defeats Odoacer at the Battle of Isonzo, forcing his way into Italy. 632 – Fatimah, daughter of the Islamic prophet Muhammad, dies, with he ...

## Reasoning-length bands

| band | reasoning tokens | control token | docs in corpus | median total tokens |
|---|---|---|---:|---:|
| **L0_direct** | 0-32 | `<effort=none>` | 7 | 77 |
| **L1_short** | 32-256 | `<effort=low>` | 17,106 | 175 |
| **L2_medium** | 256-1024 | `<effort=medium>` | 3,640 | 462 |
| **L3_long** | 1024-4096 | `<effort=high>` | 253 | 1,215 |
| **L4_ultra** | 4096-32768 | `<effort=ultra>` | 0 | 0 |

### L0_direct · `<effort=none>`

- **source** `gsm8k-plain` · **86 tokens** (31 supervised) · difficulty `B1`
  > Problem: Rocky boxed 190 fights in his career. 50 percent of his fights were knockouts, and 20 percent of the knockouts were in the first round. How many knockouts did Rocky have in the first round?
<reasoning>
Knockouts:190(.50)=95
First Round:95(.2)=19
</reasoning>
Answer: 19

### L1_short · `<effort=low>`

- **source** `gsm8k-socratic` · **342 tokens** (228 supervised) · difficulty `B1`
  > Problem: Brinley's teacher took the grade six students to the San Diego Zoo to watch and learn about animals. While at the zoo, Brinley counted 100 snakes, 80 arctic foxes, and 20 leopards. She also saw ten times more bee-eaters than leopards, half as many cheetahs as snakes, and twice as many alligators as the total number of Arctic foxes and leopards. What's the total number of animals Brinley counted at the zoo?
<reasoning>
How many more bee-eaters were there than leopards? ** There were 10*20 = <<10*20=200>>200 more bee-eaters at the zoo.
How many cheetahs were there? ** There were 1/2*100 = <<1/2*100=50>>50 cheetahs.
How many foxes and leopards were there? ** The total number of foxes a ...

### L2_medium · `<effort=medium>`

- **source** `prm800k-phase2` · **1,129 tokens** (903 supervised) · difficulty `B5`
  > Problem: A line passes through the distinct vectors $\mathbf{a}$ and $\mathbf{b}.$ Which of the following are valid parameterizations of the line?
(A) $\mathbf{a} + t (\mathbf{a} - \mathbf{b})$
(B) $\frac{3}{4} \mathbf{a} + t \cdot \frac{\mathbf{b}}{4}$
(C) $\frac{\mathbf{a} + \mathbf{b}}{2} + t \cdot \frac{\mathbf{b} - \mathbf{a}}{2}$
(D) $\mathbf{a} + 2 \mathbf{b} + t (2 \mathbf{b} - 3 \mathbf{a})$
(E) $-\mathbf{b} + t (\mathbf{a} + \mathbf{b})$
(F) $\mathbf{b} + \frac{t}{3} (2 \mathbf{a} - 2 \mathbf{b})$
Enter the letters of the correct options, separated by commas.
<reasoning>
I know that a parameterization of a line has the form $\mathbf{p} + t \mathbf{d}$, where $\mathbf{p}$ is a fixed ...

### L3_long · `<effort=high>`

- **source** `prm800k-phase2-search-reconstruction` · **3,285 tokens** (3,261 supervised) · difficulty `B5`
  > Problem: What is the largest $2$-digit prime factor of the integer $n = {200\choose 100}$?
<reasoning>
To find the largest $2$-digit prime factor of $n$, I need to first express $n$ in terms of its prime factorization.
I know that ${n\choose k} = \frac{n!}{k!(n-k)!}$, so ${200\choose 100} = \frac{200!}{100!100!}$.
This means that $n$ is the product of all the prime factors of $200!$ that are not canceled out by the prime factors of $100!100!$.
I can use the fact that $100! = 1 \times 2 \times \cdots \times 100$ and $200! = 1 \times 2 \times \cdots \times 200$ to identify which prime factors are canceled out.
For example, the prime factor $2$ appears in both $100!$ and $200!$, but it appears  ...

### L4_ultra

_No document in the cleaned corpus falls in this band. That is itself a finding: this band has to be manufactured._
