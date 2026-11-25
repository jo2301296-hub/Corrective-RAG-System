import re
from dataclasses import dataclass, field
from typing import Callable

from .config import CFG
from .generator import NOT_ANSWERABLE, generate, is_grounded
from .grader import grade
from .rewriter import rewrite
from .vectorstore import Hit, VectorStore

FALLBACK = (
    "I couldn't find a reliable answer to that in the uploaded documents, "
    "so I won't guess. Try rephrasing the question, or upload a document that covers this topic."
)


@dataclass
class Attempt:
    query: str
    hits: list[Hit]
    sufficient: bool
    feedback: str


@dataclass
class Result:
    question: str
    answer: str
    answerable: bool
    sources: list[Hit] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    reason: str = ""


def answer_question(question: str, store: VectorStore, params=CFG,
                    on_step: Callable[[str], None] | None = None) -> Result:
    log = on_step or (lambda _msg: None)

    if store.count() == 0:
        return Result(question, "No documents have been indexed yet. Upload some first.",
                      False, reason="empty index")

    attempts: list[Attempt] = []
    pool: dict[str, Hit] = {}          # chunk id -> graded, relevant chunk
    tried: list[str] = []
    query = question

    for n in range(params.max_rewrites + 1):
        tried.append(query)
        log(f"🔎 Retrieving (attempt {n + 1}): “{query}”")
        fresh = store.query(query, params.top_k)

        # grade new hits together with chunks already judged relevant earlier
        candidates = {h.id: h for h in fresh}
        candidates.update(pool)
        grading = grade(question, list(candidates.values()), params)

        pool = {h.id: h for h in grading.hits if h.score >= params.relevance_threshold}
        attempts.append(Attempt(query, grading.hits, grading.sufficient, grading.feedback))
        log(f"🧪 Graded {len(grading.hits)} chunks → {len(pool)} relevant, "
            f"sufficient: {grading.sufficient}")

        if grading.sufficient:
            break
        if n < params.max_rewrites:
            query = rewrite(question, tried, grading.feedback, params)
            log("✏️ Context is weak → rewriting the query")

    final = sorted(pool.values(), key=lambda h: -h.score)[: params.max_context_chunks]
    if not final:
        return Result(question, FALLBACK, False, attempts=attempts,
                      reason="no relevant chunks after all attempts")

    log("🧠 Generating answer from verified context")
    text = generate(question, final, params)
    if text.strip() == NOT_ANSWERABLE:
        return Result(question, FALLBACK, False, attempts=attempts,
                      reason="generator found the context insufficient")

    if params.verify_answer and not is_grounded(text, final, params):
        return Result(question, FALLBACK, False, attempts=attempts,
                      reason="answer failed the groundedness check")

    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 0 < int(m) <= len(final)})
    sources = [final[i - 1] for i in cited] or final
    return Result(question, text, True, sources, attempts)