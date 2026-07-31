"""
S5 / step 01 - acquire raw data for the lanes the V5 mixture shows to be starved.

Session 4 cleaned a 63.35M-token conversational Indic corpus. Session 5's mixture
arithmetic (see scripts/03_solve_mixture.py) says the starved lanes are agentic, reasoning,
long-context and verified-native Indic. This script fetches real, licence-checked
raw data for exactly those lanes, plus small code / web / STEM pools that the
proxy ablation needs as contrast domains.

Every fetch is recorded in data/raw/provenance.jsonl with url, licence, sha256,
byte count and UTC timestamp, continuing the S4 provenance discipline.

Hosts used are the ones reachable from this network (huggingface.co CDN is
blocked here, so nothing depends on it): dumps.wikimedia.org (cirrussearch content
dumps), raw/media.githubusercontent.com, codeload.github.com, gutenberg.org,
export.arxiv.org.

Run:  python scripts/01_fetch_lanes.py [--lane LANE ...]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PROV = RAW / "provenance.jsonl"
UA = {"User-Agent": "ERA-V5-S5-mixture-study/1.0 (course assignment; contact via github)"}

# ---------------------------------------------------------------- provenance


def record(lane: str, source: str, url: str, licence: str, path: Path, n_docs: int, note: str = "") -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rec = {
        "lane": lane,
        "source": source,
        "url": url,
        "license": licence,
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "docs": n_docs,
        "sha256": h.hexdigest(),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
    }
    with open(PROV, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [prov] {lane:14s} {source:34s} {rec['bytes']/1e6:8.2f} MB  {n_docs:>7,} docs")


def out_path(lane: str, name: str) -> Path:
    d = RAW / lane
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def already(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def get(url: str, **kw) -> requests.Response:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=kw.pop("timeout", 90), headers=UA, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


# ---------------------------------------------------------------- lane: indic verified native
# Wikipedia is the cleanest *verified-native* Indic text reachable here: written by
# native speakers in native script, versioned, attributed, CC BY-SA 4.0. It is the
# Tier-A ("verified native") anchor of the Indic split in the mixture spec.

INDIC_LANGS = {
    "hi": 6.0, "bn": 5.0, "ta": 5.0, "te": 5.0, "mr": 4.0, "gu": 3.0,
    "kn": 3.0, "ml": 3.0, "pa": 2.5, "ur": 3.0, "or": 2.0, "as": 2.0,
}  # language -> target megabytes of plain text.
# Deliberately breadth-first, not volume-first: S4 already delivered 63M tokens of
# Hindi/Hinglish. What the Indic lane is starved of is *verified-native breadth*
# across the 12 languages, which is what this pulls.

import threading  # noqa: E402
import zlib  # noqa: E402

CIRRUS = "https://dumps.wikimedia.org/other/cirrussearch/"


def _cirrus_url(lang: str) -> str | None:
    """Newest cirrussearch content dump for <lang>wiki. These carry the *rendered
    plain text* of every article, so no wikitext parsing is needed."""
    r = get(CIRRUS, timeout=90)
    dates = sorted(re.findall(r'href="(\d{8})/"', r.text))
    for date in reversed(dates[-3:]):
        listing = get(CIRRUS + date + "/", timeout=120)
        m = re.search(rf'href="({lang}wiki-\d+-cirrussearch-content\.json\.gz)"', listing.text)
        if m:
            return CIRRUS + date + "/" + m.group(1)
    return None


def fetch_wikipedia_dump(lang: str, target_mb: float, lane: str) -> None:
    """Stream a prefix of the cirrussearch dump and stop once we have enough text.

    The dump is a gzip stream of alternating {index}/{document} JSON lines. We
    decompress incrementally and abort mid-stream, which is why this takes ~1 MB
    of download per MB of text rather than the ~1 GB a full article dump costs.
    """
    path = out_path(lane, f"wiki_{lang}.jsonl")
    if already(path, int(target_mb * 1e6 * 0.75)):
        print(f"  [skip] {path.name} exists")
        return
    url = _cirrus_url(lang)
    if not url:
        print(f"  [miss] no cirrussearch dump for {lang}")
        return
    target = int(target_mb * 1e6)
    written = n = 0
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b""
    with requests.get(url, stream=True, timeout=300, headers=UA) as r, \
            open(path, "w", encoding="utf-8") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            try:
                buf += dec.decompress(chunk)
            except Exception:  # noqa: BLE001  truncated stream at the tail
                break
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                txt = (d.get("text") or "").strip()
                if len(txt) < 400:
                    continue
                f.write(json.dumps({"id": f"wiki-{lang}-{n}", "lang": lang,
                                    "title": d.get("title", ""), "text": txt},
                                   ensure_ascii=False) + "\n")
                written += len(txt.encode("utf-8"))
                n += 1
            print(f"\r    {lang}: {written/1e6:5.2f}/{target_mb:.1f} MB  {n} docs", end="")
            if written >= target:
                break
    print()
    record(lane, f"wikipedia-cirrus:{lang}", url, "CC BY-SA 4.0", path, n,
           "rendered plaintext from the cirrussearch content dump (prefix stream)")


def fetch_wikipedia(lang: str, target_mb: float, lane: str, threads: int = 4) -> None:
    path = out_path(lane, f"wiki_{lang}.jsonl")
    if already(path, int(target_mb * 1e6 * 0.75)):
        print(f"  [skip] {path.name} exists")
        return
    api = f"https://{lang}.wikipedia.org/w/api.php"
    target = int(target_mb * 1e6)
    seen: set[int] = set()
    state = {"written": 0, "docs": 0, "stall": 0}
    lock = threading.Lock()
    fh = open(path, "w", encoding="utf-8")

    def worker() -> None:
        while True:
            with lock:
                if state["written"] >= target or state["stall"] > 40:
                    return
            params = {
                "action": "query", "format": "json", "formatversion": "2",
                "generator": "random", "grnnamespace": "0", "grnlimit": "20",
                "prop": "extracts", "explaintext": "1", "exlimit": "20",
            }
            try:
                r = get(api, params=params, timeout=60)
                pages = r.json().get("query", {}).get("pages", [])
            except Exception:  # noqa: BLE001
                with lock:
                    state["stall"] += 1
                continue
            got = 0
            for p in pages:
                pid = p.get("pageid")
                txt = (p.get("extract") or "").strip()
                if len(txt) < 400:
                    continue
                with lock:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    fh.write(json.dumps({"id": f"wiki-{lang}-{pid}", "lang": lang,
                                         "title": p.get("title", ""), "text": txt},
                                        ensure_ascii=False) + "\n")
                    state["written"] += len(txt.encode("utf-8"))
                    state["docs"] += 1
                    got += 1
            with lock:
                state["stall"] = 0 if got else state["stall"] + 1
                print(f"\r    {lang}: {state['written']/1e6:5.2f}/{target_mb:.1f} MB "
                      f"{state['docs']} docs", end="")

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    fh.close()
    print()
    record(lane, f"wikipedia:{lang}", f"https://{lang}.wikipedia.org/w/api.php",
           "CC BY-SA 4.0", path, state["docs"], "random main-namespace articles, plaintext extracts")


def fetch_indic_all(max_parallel: int = 4) -> None:
    items = list(INDIC_LANGS.items())
    for i in range(0, len(items), max_parallel):
        batch = items[i:i + max_parallel]
        ts = [threading.Thread(target=fetch_wikipedia_dump, args=(l, mb, "indic_verified"))
              for l, mb in batch]
        for t in ts:
            t.start()
        for t in ts:
            t.join()


# ---------------------------------------------------------------- lane: agentic
# Gorilla (Apache-2.0) is the largest real function-calling / API-agent corpus that
# is reachable as plain files. APIBench + OpenFunctions give single and parallel
# tool calls; BFCL v4 live/multi-turn give multi-step trajectories.

GORILLA = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
AGENTIC_FILES = [
    ("openfunctions/openfunctions-v1/gorilla_openfunctions_v1_train.json", "gorilla-openfunctions-v1"),
    ("data/apibench/huggingface_train.json", "apibench-huggingface"),
    ("data/apibench/tensorflow_train.json", "apibench-tensorflow"),
    ("data/apibench/torchhub_train.json", "apibench-torchhub"),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_multiple.json", "bfcl-v4-live-multiple"),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json", "bfcl-v4-multi-turn"),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_parallel_multiple.json", "bfcl-v4-parallel"),
    # BFCL keeps the gold call sequences in a separate directory from the questions.
    # Without these the whole BFCL slice is context with zero supervised tokens.
    ("berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_live_multiple.json",
     "bfcl-v4-live-multiple.answers"),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json",
     "bfcl-v4-multi-turn.answers"),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_parallel_multiple.json",
     "bfcl-v4-parallel.answers"),
]


def fetch_agentic() -> None:
    for rel, name in AGENTIC_FILES:
        path = out_path("agentic", name + ".json")
        if already(path):
            print(f"  [skip] {path.name}")
            continue
        try:
            r = get(GORILLA + rel)
        except Exception as e:  # noqa: BLE001
            print(f"  [miss] {name}: {e}")
            continue
        path.write_bytes(r.content)
        n = r.text.count("\n") + 1
        record("agentic", name, GORILLA + rel, "Apache-2.0", path, n,
               "tool-call / API-agent data; trajectories parsed in 02_clean_lanes.py")


# ---------------------------------------------------------------- lane: reasoning
# PRM800K = 800k step-level human labels over MATH solutions: real long chains of
# thought with per-step verdicts (MIT). GSM8K train + socratic = short/medium traces.

PRM = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_train.jsonl"
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"


def fetch_reasoning(prm_mb: int = 140) -> None:
    path = out_path("reasoning", "prm800k_phase2_slice.jsonl")
    if not already(path, int(prm_mb * 1e6 * 0.5)):
        print(f"  fetching PRM800K slice (~{prm_mb} MB of {456} MB)")
        n = 0
        with requests.get(PRM, stream=True, timeout=180, headers=UA) as r, open(path, "wb") as f:
            r.raise_for_status()
            got = 0
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                print(f"\r    {got/1e6:6.1f}/{prm_mb} MB", end="")
                if got >= prm_mb * 1e6:
                    break
        print()
        # drop the final partial line
        data = path.read_bytes()
        cut = data.rfind(b"\n")
        path.write_bytes(data[: cut + 1])
        n = path.read_bytes().count(b"\n")
        record("reasoning", "prm800k-phase2", PRM, "MIT", path, n,
               f"first {prm_mb}MB slice of a 456MB file; step-level CoT with human labels")
    else:
        print("  [skip] prm800k slice")
    # test.jsonl is fetched as gsm8k_test_holdout.jsonl and is NEVER trained on - it is
    # the probe set the decontamination stage scans the corpus against.
    for name in ("train.jsonl", "train_socratic.jsonl", "test.jsonl"):
        p = out_path("reasoning", "gsm8k_test_holdout.jsonl" if name == "test.jsonl"
                     else "gsm8k_" + name)
        if already(p):
            print(f"  [skip] {p.name}")
            continue
        r = get(GSM + name)
        p.write_bytes(r.content)
        record("reasoning",
               "gsm8k-test-holdout" if name == "test.jsonl" else "gsm8k-" + name[:-6],
               GSM + name, "MIT", p, r.text.count("\n"),
               "NEVER TRAINED ON - the decontamination probe set used in stage 7"
               if name == "test.jsonl" else "short/medium worked solutions")


# ---------------------------------------------------------------- lane: long context
# Project Gutenberg public-domain books: genuine single documents of 100k+ tokens,
# which is what the long-context lane actually needs (not concatenated short docs).

GUTENBERG_IDS = [
    1342, 84, 11, 1661, 2701, 98, 1400, 174, 345, 46, 5200, 2542, 1080, 76, 16,
    2600, 1232, 4300, 2814, 158, 145, 1260, 768, 219, 1497, 30254, 6130, 3207,
    1727, 2000, 996, 43, 74, 120, 205, 209, 236, 244, 271, 289, 3600, 8800,
    1184, 100, 3296, 2554, 28054, 600, 2638, 8117, 27827, 7370, 1013, 10681,
    35, 36, 5230, 159, 164, 18857, 1998, 4363, 2680, 3825, 10007, 375, 2148,
    1946, 105, 121, 141, 161, 946, 1023, 1041, 1112, 1155, 1250, 1322, 1399,
    1404, 1513, 1524, 1597, 1656, 1695, 2147, 2265, 2591, 2852, 3090, 3155,
    3268, 3300, 3420, 3748, 4085, 4200, 4517, 5740, 6593, 7700, 8492, 9296,
    12699, 14591, 15399, 16328, 17192, 19033, 20203, 22381, 23700, 25344,
    26184, 27525, 28885, 30601, 32325, 33283, 34901, 36034, 37106, 39407,
]


def fetch_long_context(max_books: int = 120) -> None:
    path = out_path("long_context", "gutenberg.jsonl")
    have = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    have.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
    n = len(have)
    with open(path, "a", encoding="utf-8") as f:
        for bid in GUTENBERG_IDS[:max_books]:
            key = f"pg-{bid}"
            if key in have:
                continue
            for url in (f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                        f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"):
                try:
                    r = get(url, timeout=60)
                except Exception:  # noqa: BLE001
                    continue
                txt = r.text
                if len(txt) < 60_000:
                    continue
                # strip the PG header/footer so the licence boilerplate is not trained on
                s = txt.find("*** START OF")
                e = txt.find("*** END OF")
                if s != -1:
                    txt = txt[txt.find("\n", s) + 1:]
                if e != -1:
                    txt = txt[: txt.rfind("*** END OF")] if "*** END OF" in txt else txt
                txt = txt.strip()
                if len(txt) < 60_000:
                    continue
                f.write(json.dumps({"id": key, "title": f"gutenberg-{bid}", "text": txt}, ensure_ascii=False) + "\n")
                n += 1
                print(f"\r    books: {n}", end="")
                break
            time.sleep(0.3)
    print()
    record("long_context", "project-gutenberg", "https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt",
           "Public domain (US)", path, n, "full books >=60k chars, PG header/footer stripped")


# ---------------------------------------------------------------- lane: code (contrast domain)
CODE_REPOS = [
    ("psf/requests", "main", "Apache-2.0"),
    ("pallets/flask", "main", "BSD-3-Clause"),
    ("pallets/click", "main", "BSD-3-Clause"),
    ("tiangolo/fastapi", "master", "MIT"),
    ("psf/black", "main", "MIT"),
    ("numpy/numpy", "main", "BSD-3-Clause"),
    ("scikit-learn/scikit-learn", "main", "BSD-3-Clause"),
    ("karpathy/nanoGPT", "master", "MIT"),
    ("expressjs/express", "master", "MIT"),
    ("lodash/lodash", "main", "MIT"),
    ("gin-gonic/gin", "master", "MIT"),
    ("spf13/cobra", "main", "Apache-2.0"),
    ("BurntSushi/ripgrep", "master", "MIT"),
    ("sharkdp/fd", "master", "MIT"),
    ("redis/redis", "unstable", "BSD-3-Clause"),
    ("sqlite/sqlite", "master", "Public domain"),
]
CODE_EXT = (".py", ".js", ".ts", ".go", ".rs", ".c", ".h", ".cpp", ".java", ".sh", ".sql")


def fetch_code() -> None:
    path = out_path("code", "github_permissive.jsonl")
    if already(path, 20_000_000):
        print("  [skip] code corpus exists")
        return
    n = 0
    with open(path, "w", encoding="utf-8") as out:
        for repo, branch, lic in CODE_REPOS:
            url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
            try:
                r = get(url, timeout=180)
            except Exception as e:  # noqa: BLE001
                print(f"  [miss] {repo}: {e}")
                continue
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
            except zipfile.BadZipFile:
                print(f"  [miss] {repo}: bad zip")
                continue
            got = 0
            for info in z.infolist():
                if info.is_dir() or info.file_size > 200_000 or info.file_size < 400:
                    continue
                if not info.filename.endswith(CODE_EXT):
                    continue
                try:
                    txt = z.read(info).decode("utf-8")
                except Exception:  # noqa: BLE001
                    continue
                out.write(json.dumps({"id": f"{repo}:{info.filename}", "repo": repo, "license": lic,
                                      "path": info.filename, "text": txt}, ensure_ascii=False) + "\n")
                n += 1
                got += 1
            print(f"    {repo:28s} {got:5d} files")
    record("code", "github-permissive-16-repos", "https://codeload.github.com/<repo>/zip/refs/heads/<branch>",
           "MIT / Apache-2.0 / BSD-3 / public domain (per-file recorded)", path, n,
           "permissive repos only; per-file licence carried in the record")


# ---------------------------------------------------------------- lane: stem (contrast domain)
def fetch_stem(target_docs: int = 6000) -> None:
    path = out_path("stem", "arxiv_abstracts.jsonl")
    if already(path, 5_000_000):
        print("  [skip] stem corpus exists")
        return
    cats = ["math.NT", "math.AG", "math.PR", "cs.LG", "cs.CL", "physics.optics", "stat.ML", "math.CO", "cs.DS"]
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for cat in cats:
            for start in range(0, target_docs // len(cats), 100):
                url = ("http://export.arxiv.org/api/query?search_query=cat:%s&start=%d&max_results=100"
                       % (cat, start))
                try:
                    r = get(url, timeout=60)
                except Exception:  # noqa: BLE001
                    break
                entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
                if not entries:
                    break
                for e in entries:
                    title = re.search(r"<title>(.*?)</title>", e, re.S)
                    summ = re.search(r"<summary>(.*?)</summary>", e, re.S)
                    if not (title and summ):
                        continue
                    txt = (title.group(1).strip() + "\n\n" + summ.group(1).strip())
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if len(txt) < 400:
                        continue
                    f.write(json.dumps({"id": f"arxiv-{cat}-{n}", "cat": cat, "text": txt}, ensure_ascii=False) + "\n")
                    n += 1
                print(f"\r    {cat}: {n} abstracts", end="")
                time.sleep(3.1)  # arXiv API asks for >3s between calls
    print()
    record("stem", "arxiv-api-abstracts", "http://export.arxiv.org/api/query",
           "arXiv metadata (abstracts, non-exclusive licence to distribute)", path, n,
           "title+abstract only, 9 categories")


# ---------------------------------------------------------------- driver
LANES = {
    "indic": fetch_indic_all,
    "web": lambda: fetch_wikipedia_dump("en", 12.0, "web"),
    "agentic": fetch_agentic,
    "reasoning": fetch_reasoning,
    "long_context": fetch_long_context,
    "code": fetch_code,
    "stem": fetch_stem,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", choices=sorted(LANES), default=None)
    a = ap.parse_args()
    lanes = a.lane or list(LANES)
    for lane in lanes:
        print(f"\n=== fetching lane: {lane}")
        t0 = time.time()
        try:
            LANES[lane]()
        except Exception as e:  # noqa: BLE001
            print(f"  !! lane {lane} failed: {type(e).__name__}: {e}")
        print(f"  ({time.time()-t0:.0f}s)")
    print("\nprovenance ->", PROV)


if __name__ == "__main__":
    main()
