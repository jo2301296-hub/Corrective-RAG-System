from .config import CFG
from .llm import TruncatedError, complete

PROMPT = """The search query below did not retrieve enough relevant passages from a document collection.

Original user question: {question}
Queries already tried (do not repeat them):
{tried}
Evaluator feedback on what was missing: {feedback}

Write ONE improved search query that is more likely to match the wording used in documents.
Use synonyms, expand abbreviations, and focus on the key concepts and entities.
Write it as one plain phrase of at most 15 words, with no OR/AND operators or quotes.
Output only the query text, nothing else."""


def rewrite(question: str, tried: list[str], feedback: str, params=CFG) -> str:
    tried_list = "\n".join(f"- {q}" for q in tried)
    try:
        out = complete(
            PROMPT.format(question=question, tried=tried_list, feedback=feedback or "none"),
            model=params.fast_model,
            max_tokens=500,
            temperature=0.6,   # at 0 the small model can loop on synonyms forever
        )
    except TruncatedError:
        return question
    return out.strip().strip('"').splitlines()[0] if out.strip() else question
