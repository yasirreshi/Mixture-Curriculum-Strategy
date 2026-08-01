# Difficulty and reasoning-length bands, with real examples

_Every example below is a document from `data/clean/` in this repo, with the token count measured by the 32k BPE trained in `scripts/02_clean_lanes.py`. Band assignment is done by the documented heuristics in that script, not by hand._

## Difficulty ladder

| band | name | docs in corpus | lanes present | median tokens |
|---|---|---:|---|---:|
| **B0** | Nursery | 299 | general_web, indic | 2,471 |
| **B1** | Grade-school | 16,135 | code, general_web, indic, reasoning | 175 |
| **B2** | High-school | 28,020 | agentic, code, general_web, indic, reasoning | 335 |
| **B3** | Undergraduate | 5,014 | agentic, code, general_web, indic, long_context, reasoning | 479 |
| **B4** | Graduate | 3,140 | agentic, code, general_web, indic, long_context, reasoning | 1,067 |
| **B5** | Research / PhD | 7,333 | agentic, general_web, indic, reasoning, stem_math | 252 |

### B0 - Nursery

- **lane** `indic` · **source** `wikipedia:ta` · **7,542 tokens** (7,542 supervised / 0 context) · lang `ta`
  > இசை (ஒலிப்பு) (music) என்பது ஒழுங்கு செய்யப்பட்ட, கட்டுப்படுத்தப்பட்ட, அழகு ஒலியாகும். இசை என்ற சொல்லுக்கு இசைய வைப்பது என்று பொருள். மனிதனையும் மற்ற உயிரினங்களையும் இசைய வைக்கின்ற, பணிய வைக்கின்ற ஓர் அருஞ்சாதனம் இசை. இசையை சிரவண கலை எனவும் அழைப்பர். சங்கீதம் என்பது செவிக்கு இன்பம் தரும் தொனி(ஒலி)களைப் பற்றிய கலையாகும். இசையை வடமொழியில் நாதம் என அழைப்பர். இசை இன்று பல்வேறு பயன்களைத் தருகின்றது. தற்போது படித்தவர் முதல ...

- **lane** `general_web` · **source** `wikipedia:en` · **5,031 tokens** (5,031 supervised / 0 context) · lang `en`
  > Anacondas or water boas are a group of large boas of the genus Eunectes. They are a semiaquatic group of snakes found in tropical South America. Three to five extant and one extinct species are currently recognized, including one of the largest snakes in the world, E. murinus, the green anaconda. The generic name Eunectes is derived from Ancient Greek: εὐνήκτης, romanized: eunēktēs, lit. 'good swimmer'. The South Ame ...

### B1 - Grade-school

- **lane** `reasoning` · **source** `gsm8k-socratic` · **390 tokens** (287 supervised / 103 context) · lang `en`
  > Problem: Mr. Rocky went to the market to sell his handmade crafts on a particular week. He was selling jewelry at $30 each and paintings at $100 each. However, he realized the income from the sales wasn't worth the labor and materials he had used, so he decided to increase the price of jewelry by $10 each and the cost of each painting by 20%. Calculate the total price a buyer who takes two pieces of jewelry and five  ...

- **lane** `code` · **source** `sqlite/sqlite` · **191 tokens** (191 supervised / 0 context) · lang `code`
  > /*
** 2023-08-04
**
** The author disclaims copyright to this source code. In place of
** a legal notice, here is a blessing:
**
** May you do good and not evil.
** May you find forgiveness for yourself and forgive others.
** May you share freely, never taking more than you give.
**
*************************************************************************
** This file is part of the JNI bindings for the sqlite3 C API ...

### B2 - High-school

- **lane** `agentic` · **source** `gorilla-apibench-huggingface` · **521 tokens** (488 supervised / 33 context) · lang `en`
  > <user>
I am writing an automatic essay scorer. I want to identify if the given essay contradicts the provided prompt statement or not.
</user>
<assistant>
<<<domain>>>: Natural Language Processing Zero-Shot Classification
<<<api_call>>>: AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. T ...

- **lane** `code` · **source** `tiangolo/fastapi` · **602 tokens** (602 supervised / 0 context) · lang `code`
  > from fastapi import Depends, FastAPI, Security
from fastapi.security.open_id_connect_url import OpenIdConnect
from fastapi.testclient import TestClient
from inline_snapshot import snapshot
from pydantic import BaseModel
app = FastAPI()
oid = OpenIdConnect(openIdConnectUrl="/openid", auto_error=False)
class User(BaseModel):
 username: str
def get_current_user(oauth_header: str \| None = Security(oid)):
 if oauth_header ...

### B3 - Undergraduate

- **lane** `agentic` · **source** `gorilla-openfunctions-v1` · **394 tokens** (52 supervised / 342 context) · lang `en`
  > <tools>
{'name': 'PyArrow', 'api_name': 'logical_or', 'description': 'Logical OR', 'parameters': {'required': [{'name': 'x', 'description': 'Argument to compute function. Can be an array-like or scalar-like value.'}, {'name': 'y', 'description': 'Argument to compute function. Can be an array-like or scalar-like value.'}, {'name': 'memory_pool', 'description': 'Optional. If not passed, memory will be allocated from th ...

- **lane** `indic` · **source** `wikipedia:te` · **455 tokens** (455 supervised / 0 context) · lang `mixed_or_other`
  > బఠానీ లేదా బఠానీలు (శాస్త్రీయనామం: పీసం సెటైవమ్) ఒక రకమైన గింజ ధాన్యాలు. వీటిని పచ్చిగానూ, ఎండబెట్టి కూడా వాడుతారు. పచ్చి బఠానీలను కూరల వినియోగంలోనూ, ఎండు బఠానీలను చిరుతిండి గానూ ఉపయోగిస్తారు. బఠానీల తొక్కలు ముదిరి ముడతలు పడకూడదు, తొక్కలపై నల్లటి పసుపు రంగు మరకలుండరాదు Schaefer, Hanno; Hechenleitner, Paulina; Santos-Guerra, Arnoldo; de Sequeira, Miguel Menezes; Pennington, R Toby; Kenicer, Gregory; Carine, Mark A (20 ...

### B4 - Graduate

- **lane** `reasoning` · **source** `prm800k-phase2` · **301 tokens** (264 supervised / 37 context) · lang `en`
  > Problem: The ratio of the length of a rectangular room to its width is $5:3$. The perimeter of the room is 48 feet. What is the area of the room?
<reasoning>
I want to find the area of the room, which is the product of the length and the width.
But I don't know the exact values of the length and the width, only their ratio.
Maybe I can use the ratio to write the length and the width in terms of a common variable.
Let ...

- **lane** `code` · **source** `scikit-learn/scikit-learn` · **12,075 tokens** (12,075 supervised / 0 context) · lang `code`
  > """Bagging meta-estimator."""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import itertools
import numbers
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context
from sklearn.ensemble._base import BaseEnsemble, _partition_esti ...

### B5 - Research / PhD

- **lane** `stem_math` · **source** `arxiv` · **138 tokens** (138 supervised / 0 context) · lang `en`
  > Simultaneous FPQ-Ordering and Hybrid Planarity Testing We study the interplay between embedding constrained planarity and hybrid planarity testing. We consider a constrained planarity testing problem, called 1-Fixed Constrained Planarity, and prove that this problem can be solved in quadratic time for biconnected graphs. Our solution is based on a new definition of fixedness that makes it possible to simplify and ext ...

- **lane** `reasoning` · **source** `prm800k-phase2` · **366 tokens** (320 supervised / 46 context) · lang `en`
  > Problem: Alan saved 500 dollars in a bank account that compounds 3 percent annually. Assuming there are no other transactions, after 10 years, how much is in Alan's bank account? (Give your answer to the nearest dollar.)
<reasoning>
I know that compound interest means that each year, the interest is calculated based on the current balance, not the initial deposit.
So, to find the balance after 10 years, I need to app ...

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

- **source** `gsm8k-socratic` · **309 tokens** (225 supervised) · difficulty `B1`
  > Problem: The trip from Philip's house to the children's school is 2.5 miles, and the trip to the market is 2 miles. He makes the round trip to school two times (when taking the children to school and when bringing them back) every day for 4 days a week. Then he makes a round trip to the market once during weekends. What is his car's mileage for a typical week?
<reasoning>
How many miles does Philip drive when taking the children to school? ** Making the round trip to school means driving from the house to school and then back, for a total of 2.5+2.5 = <<2.5+2.5=5>>5 miles
How many miles does Philip drive when taking the children to school? ** He makes this trip twice in one day for a total o ...

### L2_medium · `<effort=medium>`

- **source** `prm800k-phase2` · **1,056 tokens** (952 supervised) · difficulty `B5`
  > Problem: Let $n$ be a positive integer. The sequence $(x_k)$ is defined by $x_0 = 0,$ $x_1 = 1,$ and
\[x_{k + 2} = \frac{(n - 1) x_{k + 1} - (n - k) x_k}{k + 1}\]for $k \ge 0.$ Find $x_0 + x_1 + x_2 + \dotsb$ as a function of $n.$
<reasoning>
I notice that the recurrence relation involves the coefficients $n - 1,$ $n - k,$ and $k + 1,$ which are all binomial coefficients.
This suggests that I might try to rewrite the recurrence in terms of binomial coefficients and see if I can simplify it.
I also notice that the initial values $x_0 = 0$ and $x_1 = 1$ are the same as the binomial coefficients $\binom{n}{0}$ and $\binom{n}{1},$ respectively.
So I wonder if I can find a pattern for $x_k$ in te ...

### L3_long · `<effort=high>`

- **source** `prm800k-phase2-search-reconstruction` · **3,285 tokens** (3,261 supervised) · difficulty `D4`
  > Problem: What is the largest $2$-digit prime factor of the integer $n = {200\choose 100}$?
<reasoning>
To find the largest $2$-digit prime factor of $n$, I need to first express $n$ in terms of its prime factorization.
I know that ${n\choose k} = \frac{n!}{k!(n-k)!}$, so ${200\choose 100} = \frac{200!}{100!100!}$.
This means that $n$ is the product of all the prime factors of $200!$ that are not canceled out by the prime factors of $100!100!$.
I can use the fact that $100! = 1 \times 2 \times \cdots \times 100$ and $200! = 1 \times 2 \times \cdots \times 200$ to identify which prime factors are canceled out.
For example, the prime factor $2$ appears in both $100!$ and $200!$, but it appears  ...

### L4_ultra

_No document in the cleaned corpus falls in this band. That is itself a finding: this band has to be manufactured._
