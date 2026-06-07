"""Generate ``node_info.csv`` for the Cora dataset.

Reads ``processed_data.pt`` from the first existing location and splits the
``raw_texts`` strings (which the project ships as ``"Title: ... Abstract: ..."``)
into ``paper_id, title, abstract`` columns.

The resulting CSV is consumed by ``cluster_prompt.load_node_titles`` and
``cluster_prompt.load_node_abstracts`` to produce *real* titles/abstracts in
prompts (instead of the long combined raw_text string, which dominates
embeddings and LLM context).

Run from the project root::

    python scripts/prepare_cora.py
    python scripts/prepare_cora.py --src cora/processed_data.pt --dst cora
    python scripts/prepare_cora.py --dst /root/autodl-tmp/cora
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd
import torch


_CANDIDATE_SRC_PATHS = [
    "cora/processed_data.pt",
    "dataset/cora/processed_data.pt",
    "/root/autodl-tmp/cora/processed_data.pt",
    "/root/autodl-tmp/dataset/cora/processed_data.pt",
]


def _resolve_src(arg_src: str | None) -> str:
    if arg_src:
        if not os.path.isfile(arg_src):
            raise FileNotFoundError(arg_src)
        return arg_src
    for c in _CANDIDATE_SRC_PATHS:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Could not locate processed_data.pt. Tried:\n  " +
        "\n  ".join(_CANDIDATE_SRC_PATHS) +
        "\nPass --src /path/to/processed_data.pt explicitly.",
    )


# ── boilerplate / submission-header patterns ────────────────────────
# Each alternative matches *one* short boilerplate clause ending in a period.
# Per-clause length is capped so a clause without a nearby period (e.g.
# "Preference: Oral presentation Reinforcement learning for cellular …")
# does NOT accidentally swallow the real title.
#
# Caps:
#   - 80 chars for clauses that legitimately can be long (conference names)
#   - 40 chars for short labels (Section / Preference / Category / …)
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"submitted\s+to\b[^.]{0,80}\.|"
    r"to\s+appear\s+(?:in|at)\b[^.]{0,80}\.|"
    r"(?:appearing|appeared)\s+(?:in|at)\b[^.]{0,80}\.|"
    r"in\s+(?:the\s+)?proceedings\s+of\b[^.]{0,80}\.|"
    r"proceedings\s+of\b[^.]{0,80}\.|"
    r"technical\s+report\b[^.]{0,80}\.|"
    r"section\s*:\s*[^.]{0,40}\.|"
    r"preference\s*:\s*[^.]{0,40}\.|"
    r"topic\s*:\s*[^.]{0,40}\.|"
    r"keywords?\s*:\s*[^.]{0,200}\.|"
    r"category\s*:\s*[^.]{0,40}\.|"
    r"contact\s+author\s*:\s*[^.]{0,80}\.|"
    r"running\s+head\s*:\s*[^.]{0,80}\.|"
    r"in\s+press\b[^.]{0,80}\.|"
    r"published\s+in\b[^.]{0,80}\."
    r")\s*",
    re.IGNORECASE,
)

_ABSTRACT_MARKER_RE = re.compile(r"(?i)\b(abstract|summary|introduction)\s*:\s*")

# Phrases that almost always signal the start of an abstract / first body
# sentence when no explicit "Abstract:" marker is present.
_ABSTRACT_OPENERS_RE = re.compile(
    r"(?i)\b("
    r"in\s+this\s+(?:paper|work|article|study|note|report)|"
    r"this\s+(?:paper|work|article|study|note|report)\b|"
    r"the\s+(?:paper|work|article|study|note|report)\s+(?:proposes|presents|describes|introduces|reports)|"
    r"we\s+(?:propose|present|describe|show|introduce|study|investigate|consider|"
    r"analyze|develop|prove|address|examine|explore|extend|generalize|argue|"
    r"demonstrate|propose\s+a|propose\s+an)\b|"
    r"recently\b"
    r")"
)


def _strip_boilerplate(s: str) -> str:
    """Peel off submission-template clauses from the start of ``s``.

    Has a safety net: if the original text contained an ``Abstract:`` /
    ``Summary:`` / ``Introduction:`` marker but the stripped text no longer
    does, we assume the regex over-ate and return the original.  Better to
    keep some boilerplate than to lose the abstract anchor.
    """
    original = s
    while True:
        m = _BOILERPLATE_RE.match(s)
        if not m:
            break
        s = s[m.end():]
    if (_ABSTRACT_MARKER_RE.search(original)
            and not _ABSTRACT_MARKER_RE.search(s)):
        return original.lstrip(" .,;:-")
    return s.lstrip(" .,;:-")


def _split_title_abstract(raw: str, title_max: int = 200) -> Tuple[str, str]:
    """Pull (title, abstract) out of strings like 'Title: foo Abstract: bar'.

    Strips common submission boilerplate (``Submitted to NIPS96, Section: …,
    Preference: …``) first, then looks for an explicit ``Abstract:`` /
    ``Summary:`` / ``Introduction:`` marker, and finally falls back to common
    abstract-opener phrases (``This paper …``, ``We propose …``).
    """
    s = str(raw or "").strip().replace("\r", " ")

    # Drop optional leading "Title:" / "title:"
    s = re.sub(r"(?i)^\s*title\s*:\s*", "", s)

    s = _strip_boilerplate(s)

    m = _ABSTRACT_MARKER_RE.search(s)
    if m:
        title = s[:m.start()].rstrip(" .;:-")
        abstract = s[m.end():].strip()
    else:
        # No explicit marker — try common abstract-opener phrases.
        # Constraints to avoid splitting mid-body:
        #   * opener must appear after ≥ 12 chars of title (not at pos 0)
        #   * opener must appear within the first 250 chars (otherwise it's
        #     almost certainly inside the body, not the abstract start)
        opener = _ABSTRACT_OPENERS_RE.search(s)
        if opener and 12 <= opener.start() <= 250:
            title = s[:opener.start()].rstrip(" .;:-")
            abstract = s[opener.start():].strip()
        else:
            # Last resort: split on the first sentence boundary.
            parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)
            title = parts[0]
            abstract = parts[1] if len(parts) > 1 else ""

    title = re.sub(r"\s+", " ", title).strip()
    abstract = re.sub(r"\s+", " ", abstract).strip()
    if len(title) > title_max:
        title = title[:title_max].rstrip() + "..."
    return title or "Unknown", abstract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="Path to processed_data.pt (auto-detected by default)")
    ap.add_argument("--dst", default="cora",
                    help="Directory to write node_info.csv (default: ./cora)")
    ap.add_argument("--title-max", type=int, default=200)
    ap.add_argument("--show-empty", type=int, default=0,
                    help="Print N rows whose abstract is empty for inspection.")
    args = ap.parse_args()

    src = _resolve_src(args.src)
    print(f"📦 Loading {src} ...")
    obj = torch.load(src, weights_only=False)

    raw_texts: List[str] | None = None
    if hasattr(obj, "raw_texts") and obj.raw_texts is not None:
        raw_texts = list(obj.raw_texts)
    elif isinstance(obj, dict):
        for k in ("raw_texts", "node_text", "texts", "title", "node_id_text"):
            if k in obj and obj[k] is not None:
                raw_texts = list(obj[k])
                break
    if raw_texts is None:
        raise RuntimeError(
            "processed_data.pt has no `raw_texts` list — cannot derive titles."
        )

    rows = []
    raw_lens: List[int] = []
    n_with_abstract = 0
    for i, raw in enumerate(raw_texts):
        raw_lens.append(len(str(raw or "")))
        title, abstract = _split_title_abstract(raw, title_max=args.title_max)
        if abstract:
            n_with_abstract += 1
        rows.append({"paper_id": i, "title": title, "abstract": abstract})

    os.makedirs(args.dst, exist_ok=True)
    out_path = os.path.join(args.dst, "node_info.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(
        f"✅ Wrote {len(rows)} rows to {out_path}\n"
        f"   {n_with_abstract} rows have a non-empty abstract "
        f"({n_with_abstract / max(1, len(rows)):.1%})"
    )

    # Quick sanity peek
    head = pd.DataFrame(rows[:3])
    print("\n  Sample:")
    for _, r in head.iterrows():
        print(f"   id={r['paper_id']}  title={r['title'][:80]!r}  "
              f"abstract={r['abstract'][:60]!r}{'...' if r['abstract'] else ''}")

    if args.show_empty > 0:
        # Length distribution of empty-abstract raw_texts: helps decide whether
        # those rows are *truly* title-only or our regexes are missing them.
        empty_idx = [r["paper_id"] for r in rows if not r["abstract"]]
        empty_lens = [raw_lens[i] for i in empty_idx]
        if empty_lens:
            short = sum(1 for n in empty_lens if n <= 200)
            long_ = sum(1 for n in empty_lens if n > 500)
            print(
                f"\n  raw_text length stats for {len(empty_lens)} empty-abstract rows: "
                f"min={min(empty_lens)}  median={sorted(empty_lens)[len(empty_lens)//2]}  "
                f"max={max(empty_lens)}  ≤200chars={short}  >500chars={long_}"
            )

        print(f"\n  First {min(args.show_empty, len(empty_idx))} samples "
              f"with empty abstract (raw_len = full raw_text length):")
        for pid in empty_idx[: args.show_empty]:
            row = rows[pid]
            t = row["title"][:120].replace("\n", " ")
            n = raw_lens[pid]
            preview = str(raw_texts[pid] or "").replace("\n", " ")[:160]
            print(f"   id={pid}  raw_len={n}  title={t!r}")
            if n > len(row["title"]) + 10:
                print(f"      raw[:160]={preview!r}")


if __name__ == "__main__":
    main()
