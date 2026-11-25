import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    llm_model: str = os.getenv("CRAG_LLM_MODEL", "openai/gpt-oss-120b")   # answer generation
    fast_model: str = os.getenv("CRAG_FAST_MODEL", "openai/gpt-oss-20b")  # grading / rewriting

    max_output_tokens: int = 2048      # kept low: free/dev tiers limit tokens per minute

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "chroma_db"
    collection: str = "documents"

    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 40

    top_k: int = 6
    relevance_threshold: float = 0.6   # chunk kept if grader score >= this
    min_relevant_chunks: int = 1       # needed to call the context "sufficient"
    max_rewrites: int = 2              # query rewrites after the first attempt
    max_context_chunks: int = 5        # chunks passed to the generator
    verify_answer: bool = True         # groundedness check after generation


CFG = Config()