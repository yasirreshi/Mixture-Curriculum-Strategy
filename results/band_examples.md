# Difficulty and reasoning-length bands, with real examples

_Every example below is a document from `data/clean/` in this repo, with the token count measured by the 32k BPE trained in `scripts/02_clean_lanes.py`. Band assignment is done by the documented heuristics in that script, not by hand._

## Difficulty ladder

| band | name | docs in corpus | lanes present | median tokens |
|---|---|---:|---|---:|
| **D0** | Nursery | 375 | general_web, indic | 2,547 |
| **D1** | School | 16,442 | code, general_web, indic, reasoning | 177 |
| **D2** | High school / undergrad | 29,302 | agentic, code, general_web, indic, long_context, reasoning | 346 |
| **D3** | Graduate | 11,830 | agentic, code, general_web, indic, long_context, reasoning, stem_math | 288 |
| **D4** | Frontier / PhD | 1,992 | agentic, reasoning | 581 |

### D0 - Nursery

- **lane** `indic` · **source** `wikipedia:or` · **341 tokens** (341 supervised / 0 context) · lang `mixed_or_other`
  > ଇଂରାଜୀ ଲିଙ୍କ https://or.wikipedia.org/wiki/Sunday ରବିବାର ହେଲା ସପ୍ତାହର ଏମିତି ଏକ ଦିନ ଯାହା ଶନିବାର ଆଉ ସୋମବାର ମଝିରେ ଆସେ । ରବିବାର ହେଉଛି ସୂର୍ଯ୍ୟଦେବଙ୍କ ବାର । ଏହିଦିନ ସୂର୍ଯ୍ୟଙ୍କୁ ଧ୍ୟାନ କରି ବ୍ରତ ରଖିବା ସହିତ ସାତ୍ତ୍ୱିକ ଲୁଣବିହୀନ ଖାଦ୍ୟ ଗ୍ରହଣ କରିବା ଉଚିତ । ପୃଥିବୀସାରା ଏହା ଏକ ସାଧାରଣ ଛୁଟିଦିନ । ତେବେ ଗଲ୍ଫ/ଆରବି ଦେଶଗୁଡ଼ିକରେ ଏହି ଦିନରେ ଅଫିସସବୁ ଖୋଲା ରହେ । "Which is the first day of the week? And which is week 1 of the year? (FAQ - Time) : FAQs  ...

- **lane** `general_web` · **source** `wikipedia:en` · **440 tokens** (440 supervised / 0 context) · lang `en`
  > Algorithms is a monthly peer-reviewed open-access scientific journal of mathematics, covering design, analysis, and experiments on algorithms. The journal is published by MDPI and was established in 2008. The founding editor-in-chief was Kazuo Iwama (Kyoto University). From May 2014 to September 2019, the editor-in-chief was Henning Fernau (Universität Trier). The current editor-in-chief is Frank Werner (Otto-von-Gue ...

### D1 - School

- **lane** `reasoning` · **source** `gsm8k-socratic` · **125 tokens** (71 supervised / 54 context) · lang `en`
  > Problem: Of the 24 students in class, one-third are in the after-school chess program, and half of those students will be absent from school on Friday for the regional chess tournament. How many of the students in this class are going to that tournament?
<reasoning>
How many students are in the after-school chess program? ** There are 24/3=<<24/3=8>>8 students in the program.
How many students are going to the tourna ...

- **lane** `code` · **source** `numpy/numpy` · **261 tokens** (261 supervised / 0 context) · lang `code`
  > #include <immintrin.h>
/**
 * Test BW mask operations due to:
 * - MSVC has supported it since vs2019 see,
 * https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
 * - Clang >= v8.0
 * - GCC >= v7.1
 */
int main(void)
{
 __mmask64 m64 = _mm512_cmpeq_epi8_mask(_mm512_set1_epi8((char)1), _mm512_set1_epi8((char)1));
 m64 = _kor_mask64(m64, m64);
 m64 = _kxor_mask64(m64 ...

### D2 - High school / undergrad

- **lane** `code` · **source** `scikit-learn/scikit-learn` · **898 tokens** (898 supervised / 0 context) · lang `code`
  > """
==========================================================
Demonstrating the different strategies of KBinsDiscretizer
==========================================================
This example presents the different strategies implemented in KBinsDiscretizer:
- 'uniform': The discretization is uniform in each feature, which means that
 the bin widths are constant in each dimension.
- 'quantile': The discretization i ...

- **lane** `agentic` · **source** `gorilla-openfunctions-v1` · **373 tokens** (73 supervised / 300 context) · lang `en`
  > <tools>
{'name': 'RapidAPI', 'api_name': 'requests.get', 'description': 'This ready-to-use API provides high-accuracy brand detection and logo recognition. [![Examples](https://img.shields.io/badge/View%20examples-gray?logo=gitlab&style=flat)](https://gitlab.com/api4ai/examples/brand-det) [![API4AI](https://img.shields.io/badge/api4.ai%20platform-fee33c?logo=icloud&flat&logoColor=black)](https://api4.ai/apis/brand-re ...

### D3 - Graduate

- **lane** `agentic` · **source** `gorilla-openfunctions-v1` · **607 tokens** (36 supervised / 571 context) · lang `en`
  > <tools>
{'name': 'active-directory', 'api_name': 'gcloud.active_directory.domains().restore(domain="my_domain.com", backup="my_backup"', 'description': 'Restore a Managed Microsoft AD domain', 'parameters': [{'name': 'domain'}, {'name': 'backup'}, {'name': 'async'}]}
{'name': 'RapidAPI', 'api_name': 'requests.get', 'description': "Do you care about accessing free images of dogs on the internet? Do you want your dog p ...

- **lane** `stem_math` · **source** `arxiv` · **481 tokens** (481 supervised / 0 context) · lang `en`
  > A Maxwell principle for generalized Orlicz balls In [A dozen de {F}inetti-style results in search of a theory, Ann. Inst. H. Poincaré Probab. Statist. 23(2)(1987), 397--423], Diaconis and Freedman studied low-dimensional projections of random vectors from the Euclidean unit sphere and the simplex in high dimensions, noting that the individual coordinates of these random vectors look like Gaussian and exponential rand ...

### D4 - Frontier / PhD

- **lane** `reasoning` · **source** `prm800k-phase2` · **844 tokens** (752 supervised / 92 context) · lang `en`
  > Problem: Let $z_1$, $z_2$, $z_3$, $\dots$, $z_{12}$ be the 12 zeroes of the polynomial $z^{12} - 2^{36}$. For each $j$, let $w_j$ be one of $z_j$ or $iz_j$. Find the maximum possible value of the real part of
\[\sum_{j = 1}^{12} w_j.\]
<reasoning>
I recognize that $z^{12} - 2^{36}$ is a difference of two perfect 12th powers, so I can factor it as $(z^6 - 2^{18})(z^6 + 2^{18})$.
Then, each factor is a difference or su ...

- **lane** `agentic` · **source** `bfcl-v4-multi-turn` · **638 tokens** (286 supervised / 352 context) · lang `en`
  > <user>
I need to schedule a flight soon on booking id 1234 and want to examine different alternatives. Calculate the cost of a business class ticket for the first airport on the available list to the last airport on the same list, on the last day of October 2026. Provide the estimated total payable amount for me.
</user>
<tool_call>
list_all_airports()
get_flight_cost(travel_from='RMS', travel_to='BOS', travel_date=' ...

## Reasoning-length bands

| band | reasoning tokens | control token | docs in corpus | median total tokens |
|---|---|---|---:|---:|
| **L0_direct** | 0-32 | `<effort=none>` | 7 | 77 |
| **L1_short** | 32-256 | `<effort=low>` | 17,106 | 175 |
| **L2_medium** | 256-1024 | `<effort=medium>` | 3,640 | 462 |
| **L3_long** | 1024-4096 | `<effort=high>` | 0 | 0 |
| **L4_ultra** | 4096-32768 | `<effort=ultra>` | 0 | 0 |

### L0_direct · `<effort=none>`

- **source** `gsm8k-plain` · **86 tokens** (31 supervised) · difficulty `D1`
  > Problem: Rocky boxed 190 fights in his career. 50 percent of his fights were knockouts, and 20 percent of the knockouts were in the first round. How many knockouts did Rocky have in the first round?
<reasoning>
Knockouts:190(.50)=95
First Round:95(.2)=19
</reasoning>
Answer: 19

### L1_short · `<effort=low>`

- **source** `gsm8k-socratic` · **365 tokens** (235 supervised) · difficulty `D1`
  > Problem: Andrew is having two of his friends over for a sleepover. For one friend, Brian, he asks his mother to buy 3 donuts. He asks for his other friend, Samuel, to get the same. A day before the sleepover Andrew ends up inviting two more friends and asks his mother to buy them the same amount of donuts. Just in case, Andrew’s mother wants to buy one more donut for each of Andrew’s friends. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else. How many donuts will Andrew’s mother need to buy?
<reasoning>
How many donuts will Andrew’s friends get? ** Andrew is having 4 friends over who will get 3 donuts each, 4 x 3 = <<4*3=12>>12 donuts total.
How many donu ...

### L2_medium · `<effort=medium>`

- **source** `prm800k-phase2` · **971 tokens** (930 supervised) · difficulty `D4`
  > Problem: Given $\tan \theta = 5,$ find
\[\frac{1 - \cos \theta}{\sin \theta} - \frac{\sin \theta}{1 + \cos \theta}.\]
<reasoning>
To solve this problem, I need to express the cosine and sine of $\theta$ in terms of the tangent.
I recall that $\tan \theta = \frac{\sin \theta}{\cos \theta},$ so I can solve for $\cos \theta$ by multiplying both sides by $\cos \theta$ and dividing by $\tan \theta.$
This gives me $\cos \theta = \frac{\sin \theta}{\tan \theta}.$
I also recall that $\sin^2 \theta + \cos^2 \theta = 1,$ so I can use this identity to find $\sin \theta.$
Substituting $\cos \theta = \frac{\sin \theta}{\tan \theta}$ into the identity, I get $\sin^2 \theta + \frac{\sin^2 \theta}{\tan^2 \t ...

### L3_long

_No document in the cleaned corpus falls in this band. That is itself a finding: this band has to be manufactured._

### L4_ultra

_No document in the cleaned corpus falls in this band. That is itself a finding: this band has to be manufactured._
