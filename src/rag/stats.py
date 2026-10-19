"""Sanity report over the chunk files: are the chunks the size and shape we expect?"""

import json
from collections import Counter

from rag.clean import is_bib_like
from rag.settings import Settings

BINS = [64, 128, 256, 384, 512]


def report(s: Settings) -> str:
    papers = sorted(p.stem for p in s.md_dir.glob("*.json") if not p.stem.startswith("_"))
    per_paper: dict[str, int] = {}
    tokens: list[int] = []
    types, tops = Counter(), Counter()
    with_abstract, no_path, bib_left = set(), 0, 0

    for pid in papers:
        f = s.chunks_dir / f"{pid}.jsonl"
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []
        per_paper[pid] = len(recs)
        for r in recs:
            tokens.append(r["n_tokens"])
            types[r["section_type"]] += 1
            tops[r["section_top"]] += 1
            if r["section_type"] == "abstract":
                with_abstract.add(pid)
            elif not r["section_path"]:
                no_path += 1
            paras = r["text"].split("\n\n")
            if sum(map(is_bib_like, paras)) >= max(2, len(paras) / 2):
                bib_left += 1

    n = len(tokens) or 1
    missing = [p for p in papers if not (s.chunks_dir / f"{p}.jsonl").exists()]
    empty = [p for p, c in per_paper.items() if c == 0 and p not in missing]
    hist = Counter(next((f"<{b}" for b in BINS if t < b), f">={BINS[-1]}") for t in tokens)
    pct = lambda k: f"{100 * k / n:.1f}%"  # noqa: E731

    return "\n".join(
        [
            f"papers            {len(papers)}  (not chunked: {len(missing)}, zero chunks: {len(empty)} {empty[:5]})",
            f"chunks            {len(tokens)}  ({len(tokens) / max(1, len(papers) - len(missing)):.1f} per paper, "
            f"{sum(tokens) / 1e6:.2f}M tokens, mean {sum(tokens) / n:.0f})",
            f"abstract chunk    {len(with_abstract)} / {len(papers)} papers",
            f"over max tokens   {sum(t > s.chunk_max_tokens for t in tokens)}  (max {max(tokens, default=0)})",
            "token histogram   " + "  ".join(f"{k}: {pct(hist[k])}" for k in [*(f'<{b}' for b in BINS), f'>={BINS[-1]}']),
            "section types     " + "  ".join(f"{k} {v}" for k, v in types.most_common()),
            f"no section path   {no_path}",
            f"bib-like left     {bib_left} ({pct(bib_left)})",
            "top sections      " + ", ".join(f"{k} {v}" for k, v in tops.most_common(15)),
        ]
    )
