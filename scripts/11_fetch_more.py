"""
S5 / step 11 - collection aimed at the slots that starve the *experiment*.

Step 01 collected for the lanes the mixture starves: agentic, reasoning, long-context and
verified-native Indic. That was the right target for the plan. Running the proxy
epoch-honestly on GPU exposed a second, inverted scarcity.

The epoch-honest proxy caps each lane's stream at (real unique supply x proxy budget / 2.9T),
so an arm that overspends a scarce lane genuinely repeats it. That construction only holds
while the local corpus can actually fill the cap. Measured against our cleaned corpus:

    lane            available     modelled supply    max epoch-honest budget
    general_web      3.51M                4.5T                  2.26M   <-- binding
    stem_math        1.10M                250B                 12.73M
    code            17.41M                1.1T                 45.91M
    indic           13.64M                276B                143.36M
    reasoning        5.34M                 85B                182.23M
    long_context    28.59M                100B                829.02M
    agentic         11.78M               10.6B               3214.25M

The lanes with the *most* data in the world are the ones our corpus holds least of relative
to their supply, so they, not agentic, cap the experiment at 2.26M tokens. Collecting for
the run and collecting for the experiment that validates the run are different problems.

This step raises the two binding lanes so the proxy can run at a ~20M-token budget with the
plan's real epoch ratios intact, and adds the tokens to the cumulative cleaning total.

Run: python scripts/11_fetch_more.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# reuse step 01's fetchers verbatim - same provenance logging, same licence recording
spec = importlib.util.spec_from_file_location("fetch01", ROOT / "scripts" / "01_fetch_lanes.py")
f01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f01)

# how much each lane needs for a 20M-token epoch-honest budget, with headroom
WEB_TARGET_MB = 170.0          # -> ~40M clean tokens against the 31M needed
STEM_CATS = ["math.NT", "math.AG", "math.PR", "cs.LG", "cs.CL", "physics.optics",
             "stat.ML", "math.CO", "cs.DS", "math.DG", "math.ST", "cs.CC", "cs.IT",
             "q-bio.NC", "physics.comp-ph", "math.RA", "cs.CV", "econ.EM"]
STEM_PER_CAT = 1600


def fetch_web():
    """Re-stream the English cirrussearch dump to a much longer prefix.

    Same function step 01 used, same URL, same licence - only the target length changes.
    """
    path = RAW / "web" / "wiki_en.jsonl"
    before = path.stat().st_size if path.exists() else 0
    print(f"general_web: {before/1e6:.1f} MB -> target {WEB_TARGET_MB:.0f} MB")
    f01.fetch_wikipedia_dump("en", WEB_TARGET_MB, "web")
    after = path.stat().st_size
    print(f"  {before/1e6:.1f} MB -> {after/1e6:.1f} MB")


def fetch_stem():
    """More arXiv abstracts, across 18 categories instead of 9.

    Written to its own file so step 01's output is left untouched and the two are
    distinguishable in the manifest. `02_clean_lanes.py` globs data/raw/stem/*.jsonl,
    so the new file is picked up with no change to the cleaning pass.
    """
    path = RAW / "stem" / "arxiv_abstracts_v2.jsonl"
    seen = set()
    old = RAW / "stem" / "arxiv_abstracts.jsonl"
    if old.exists():
        for line in open(old, encoding="utf-8"):
            try:
                seen.add(json.loads(line)["text"][:120])
            except Exception:  # noqa: BLE001
                pass
    print(f"stem_math: {len(seen)} abstracts already held; pulling {len(STEM_CATS)} categories")
    n = dup = 0
    with open(path, "w", encoding="utf-8") as f:
        for cat in STEM_CATS:
            got = 0
            for start in range(0, STEM_PER_CAT, 100):
                url = ("http://export.arxiv.org/api/query?search_query=cat:%s"
                       "&start=%d&max_results=100" % (cat, start))
                try:
                    r = f01.get(url, timeout=60)
                except Exception as e:  # noqa: BLE001
                    print(f"    {cat}: stopped ({type(e).__name__})")
                    break
                entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
                if not entries:
                    break
                for e in entries:
                    title = re.search(r"<title>(.*?)</title>", e, re.S)
                    summ = re.search(r"<summary>(.*?)</summary>", e, re.S)
                    if not (title and summ):
                        continue
                    txt = re.sub(r"\s+", " ",
                                 title.group(1).strip() + "\n\n" + summ.group(1).strip()).strip()
                    if len(txt) < 400:
                        continue
                    if txt[:120] in seen:
                        dup += 1
                        continue
                    seen.add(txt[:120])
                    f.write(json.dumps({"id": f"arxiv2-{cat}-{n}", "cat": cat, "text": txt},
                                       ensure_ascii=False) + "\n")
                    n += 1
                    got += 1
                time.sleep(3.0)      # arXiv API asks for one request per 3s
                print(f"\r    {cat:16s} +{got:>5}  total {n:>6}  (dup {dup})", end="")
            print()
    f01.record("stem", "arxiv-api:abstracts-v2",
               "http://export.arxiv.org/api/query", "arXiv terms of use", path, n,
               f"{len(STEM_CATS)} categories, deduplicated against the step-01 pull")
    print(f"  wrote {n:,} new abstracts ({path.stat().st_size/1e6:.1f} MB), {dup} duplicates skipped")


if __name__ == "__main__":
    t0 = time.time()
    fetch_web()
    fetch_stem()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> now re-run scripts/02_clean_lanes.py")
