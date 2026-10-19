"""Stage 5: retrieval evaluation.

  rag eval-gen   the local LLM writes one question per random chunk -> data/eval/synthetic.jsonl
                 (relevance: the source chunk = 2, its neighbouring chunks = 1)
  you            add hand-written questions to data/eval/manual.jsonl:
                 {"question": "...", "paper_id": "2610.13573", "pages": [2, 3]}   ("pages" optional)
  rag eval       run every configuration on every question; score the rankings at two levels:
                   chunk level: did we find the exact passage?  (questions with chunk labels)
                   paper level: did we find the right paper?    (all questions)
"""

import json
import random
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from rag.files import atomic_write
from rag.metrics import METRICS
from rag.settings import Settings

CONFIGS = ("dense", "bm25", "hybrid", "hybrid+rerank", "full")  # full = rewrite + filter + multi-query + rerank
DEPTH = 20  # chunks retrieved per question; the paper ranking is derived from these

GEN_PROMPT = """You write evaluation questions for a search engine over thousands of arXiv AI papers.
Given one passage, write ONE question that this passage answers.
- Ask like a researcher who has not read the passage: paraphrase, do not copy its phrases.
- Include enough specifics (method, dataset, system or task names) that the question points to this passage rather than to papers in general.
- Do not mention "the passage", "the paper", "the authors" or "this study", and do not quote the paper's title.
Reply with JSON only."""


class GeneratedQuestion(BaseModel):
    question: str


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{path.name} line {n} is not a complete JSON object ({e.msg}).\n"
                "JSONL = exactly one {...} per line: no line breaks inside an object, no commas between lines, no [ ]."
            ) from None
    return rows


def _chunks(s: Settings, pid: str) -> list[dict]:
    return _read_jsonl(s.chunks_dir / f"{pid}.jsonl")


def unload_llm(s: Settings, model: str) -> None:
    """An empty request with keep_alive=0 makes Ollama free the model's VRAM now."""
    from ollama import Client

    Client(host=s.ollama_url).generate(model=model, keep_alive=0)


# ---------------------------------------------------------------- test set


def generate_questions(s: Settings, n: int = 100, seed: int = 0) -> tuple[Path, int]:
    """Synthetic test set: the answer location is known because the question was written FROM that chunk."""
    from ollama import Client

    client = Client(host=s.ollama_url)
    rng = random.Random(seed)  # fixed seed: the same chunks every time, so results are comparable
    pids = sorted(p.stem for p in s.chunks_dir.glob("*.jsonl"))
    out, seen, rejected = [], set(), 0
    with tqdm(total=n, unit="question") as bar:
        while len(out) < n:
            pid = rng.choice(pids)
            recs = _chunks(s, pid)
            r = rng.choice(recs)
            if r["chunk_id"] in seen or r["n_tokens"] < 120 or r["text"].count("|") > 20:
                continue  # tiny fragments and table dumps make poor questions
            seen.add(r["chunk_id"])
            reply = client.chat(
                model=s.rewrite_model,
                messages=[
                    {"role": "system", "content": GEN_PROMPT},
                    {
                        "role": "user",
                        "content": f"Paper: {r['title']}\nSection: {' > '.join(r['section_path'])}\n\n{r['text']}",
                    },
                ],
                format=GeneratedQuestion.model_json_schema(),
                think=False,
                keep_alive="10m",  # keep the LLM loaded for the whole batch
                options={"temperature": 0.7, "num_ctx": 4096},
            )
            try:
                question = GeneratedQuestion.model_validate_json(reply.message.content).question.strip()
            except ValidationError:
                question = ""
            if len(question.split()) < 5:
                rejected += 1
                continue
            neighbours = {c["chunk_id"]: 1 for c in recs if abs(c["chunk_index"] - r["chunk_index"]) == 1}
            out.append(
                {
                    "id": f"syn-{len(out) + 1:04d}",
                    "origin": "synthetic",
                    "question": question,
                    "paper_id": pid,
                    "relevant": neighbours | {r["chunk_id"]: 2},
                }
            )
            bar.update()
    unload_llm(s, s.rewrite_model)
    s.eval_dir.mkdir(parents=True, exist_ok=True)
    path = s.eval_dir / "synthetic.jsonl"
    atomic_write(path, "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out))
    return path, rejected


def load_questions(s: Settings) -> list[dict]:
    """Synthetic questions + hand-written ones. A manual question with "pages" gets chunk-level labels:
    every chunk of that paper overlapping those pages; without pages it is scored at paper level only."""
    questions = _read_jsonl(s.eval_dir / "synthetic.jsonl")
    for i, m in enumerate(_read_jsonl(s.eval_dir / "manual.jsonl"), 1):
        chunks = _chunks(s, m["paper_id"])
        if not chunks:  # a wrong label silently makes the system look worse, so refuse it
            raise ValueError(f"manual.jsonl line {i}: paper_id {m['paper_id']!r} has no chunks (typo?)")
        relevant = {}
        if pages := m.get("pages"):
            lo, hi = min(pages), max(pages)
            relevant = {c["chunk_id"]: 1 for c in chunks if c["page_start"] <= hi and c["page_end"] >= lo}
            if not relevant:
                raise ValueError(f"manual.jsonl line {i}: no chunk of {m['paper_id']} covers pages {pages}")
        questions.append(
            {"id": f"man-{i:03d}", "origin": "manual", "question": m["question"], "paper_id": m["paper_id"], "relevant": relevant}
        )
    return questions


# ---------------------------------------------------------------- scoring


def paper_ranking(chunk_ids: list[str]) -> list[str]:
    """Chunk ranking -> paper ranking (first appearance wins). chunk_id = "<paper_id>:<index>"."""
    return list(dict.fromkeys(c.rsplit(":", 1)[0] for c in chunk_ids))


def score(questions: list[dict], runs: dict[str, dict[str, list[str]]], k: int) -> tuple[str, dict]:
    """Average every metric over the questions, per configuration, at chunk and paper level."""
    levels = {
        "chunk": [(q, q["relevant"]) for q in questions if q["relevant"]],
        "paper": [(q, {q["paper_id"]: 1}) for q in questions],
    }
    lines, summary = [], {}
    for level, items in levels.items():
        if not items:
            continue
        lines.append(f"\n{level.upper()} level: {len(items)} questions, top k = {k}")
        lines.append(f"{'config':<15}" + "".join(f"{name + '@' + str(k):>9}" for name in METRICS))
        for cfg, ranked in runs.items():
            totals = dict.fromkeys(METRICS, 0.0)
            for q, rel in items:
                r = ranked[q["id"]] if level == "chunk" else paper_ranking(ranked[q["id"]])
                for name, fn in METRICS.items():
                    totals[name] += fn(r, rel, k)
            avg = {name: total / len(items) for name, total in totals.items()}
            summary[f"{level}/{cfg}"] = avg
            lines.append(f"{cfg:<15}" + "".join(f"{v:>9.3f}" for v in avg.values()))
    return "\n".join(lines), summary


def run(s: Settings, k: int = 5, configs=CONFIGS, limit: int | None = None) -> str:
    from rag.answer import retrieve
    from rag.query import plan_query
    from rag.search import search

    questions = load_questions(s)
    questions = questions[:limit] if limit else questions
    if not questions:
        raise FileNotFoundError(f"no questions in {s.eval_dir}: run `rag eval-gen` and/or write manual.jsonl")

    plans = {}
    if "full" in configs:  # do every LLM rewrite first, with the LLM loaded once; then free the GPU for retrieval
        s_rewrite = s.model_copy(update={"ollama_keep_alive": "10m"})
        for q in tqdm(questions, desc="rewrite", unit="q"):
            plans[q["id"]] = plan_query(s_rewrite, q["question"], rewrite=True)
        unload_llm(s, s.rewrite_model)

    depth = max(k, DEPTH)
    runs: dict[str, dict[str, list[str]]] = {}
    for cfg in configs:
        ranked = {}
        for q in tqdm(questions, desc=cfg, unit="q"):
            if cfg in ("dense", "bm25"):
                payloads = [h.payload for h in search(s, q["question"], mode=cfg, k=depth)]
            else:
                plan = plans[q["id"]] if cfg == "full" else plan_query(s, q["question"], rewrite=False)
                payloads = [p for p, _ in retrieve(s, plan, rerank=cfg != "hybrid", top_k=depth)]
            ranked[q["id"]] = [p["chunk_id"] for p in payloads]
        runs[cfg] = ranked

    report, summary = score(questions, runs, k)
    out_dir = s.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    atomic_write(out, json.dumps({"k": k, "summary": summary, "questions": questions, "runs": runs}, indent=1))
    return report + f"\n\nper-question rankings saved to {out}"
