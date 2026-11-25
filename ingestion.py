import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CFG


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


# ---------- 1. Extract ----------
def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. page_number is 0 when the format has no pages."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return [(0, "\n\n".join(p.text for p in doc.paragraphs))]
    if ext in {"txt", "md"}:
        return [(0, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: .{ext}")


# ---------- 2. Clean ----------
def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)           # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)                 # collapse spaces/tabs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # single newline -> space
    text = re.sub(r"\n{3,}", "\n\n", text)              # collapse blank lines
    return text.strip()


# ---------- 3. Split ----------
def split_pages(filename: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.chunk_size,
        chunk_overlap=CFG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    idx = 0
    for page_no, raw in pages:
        text = clean_text(raw)
        if not text:
            continue
        for piece in splitter.split_text(text):
            if len(piece) < CFG.min_chunk_chars:
                continue
            cid = hashlib.sha1(f"{filename}|{page_no}|{idx}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, piece, {"source": filename, "page": page_no, "chunk_index": idx}))
            idx += 1
    return chunks


# ---------- Orchestrator ----------
def ingest_file(store, filename: str, data: bytes) -> int:
    """Extract -> clean -> split -> embed -> store. Re-uploading a file replaces it."""
    pages = extract_pages(filename, data)
    chunks = split_pages(filename, pages)
    if not chunks:
        raise ValueError(
            f"No extractable text in '{filename}'. If it is a scanned PDF, run OCR first."
        )
    store.delete_source(filename)
    store.add(chunks)
    return len(chunks)