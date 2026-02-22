"""
Kinesis Reactivation — Knowledge Vault (RAG) with FAISS + Mistral AI embeddings.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import VAULT_DIR
from src.llm.providers import get_embeddings


def load_vault():
    index_path = os.path.join(VAULT_DIR, "faiss_index")
    emb = get_embeddings()
    if emb is None:
        return None
    if os.path.exists(index_path) and os.path.isdir(index_path):
        try:
            # SAFETY: allow_dangerous_deserialization is required for FAISS pickle loading.
            # These index files are created locally by this application only.
            return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[WARN] Failed to load FAISS index: {e}")
    return None


def add_docs_to_vault(docs: list) -> bool:
    emb = get_embeddings()
    if not docs or emb is None:
        return False
    index_path = os.path.join(VAULT_DIR, "faiss_index")
    try:
        if os.path.exists(index_path) and os.path.isdir(index_path):
            # SAFETY: Index files created locally by this application only.
            v = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
            v.add_documents(docs)
        else:
            v = FAISS.from_documents(docs, emb)
        v.save_local(index_path)
        return True
    except Exception as e:
        print(f"[WARN] Failed to update FAISS vault: {e}")
        return False


def file_to_docs(uploaded_file, source_name: str) -> list:
    from pypdf import PdfReader
    if uploaded_file is None:
        return []
    raw = uploaded_file.read()
    name = (uploaded_file.name or "").lower()
    docs = []
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(BytesIO(raw))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text.strip(), metadata={"source": source_name, "page": i + 1}))
        elif name.endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
            docs.append(Document(page_content=text, metadata={"source": source_name}))
        return docs
    except Exception:
        return []


def chunk_docs(docs: list, chunk_size: int = 800, overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " "])
    out = []
    for d in docs:
        out.extend(splitter.split_documents([d]))
    return out


def vault_retrieve(query: str, k: int = 8, filter_source: Optional[str] = None) -> List[dict]:
    """Return list of {"content", "source", "page", "score"} for schema alignment."""
    v = load_vault()
    if v is None:
        return []
    try:
        results = v.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            if filter_source and doc.metadata.get("source") != filter_source:
                continue
            src = doc.metadata.get("source", "vault")
            page = doc.metadata.get("page", 0)
            if page == "":
                page = 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0
            out.append({
                "content": doc.page_content,
                "source": src,
                "page": page,
                "score": float(score),
            })
        return out
    except Exception:
        return []


def format_citations(chunks: List[dict]) -> str:
    """Format chunk list for prompt (chunks have content, source, page, score)."""
    lines = []
    for c in chunks:
        content = c.get("content", "")
        src = c.get("source", "vault")
        page = c.get("page", 0)
        cite = f"[{src}" + (f", page {page}" if page else "") + "]"
        quote = (content[:500] + "\u2026") if len(content) > 500 else content
        lines.append(f"As per {cite}: {quote}")
    return "\n\n".join(lines) if lines else ""


def citations_to_schema(chunks: List[dict]) -> List[dict]:
    """Writer output schema: list of {source, page, score}."""
    return [{"source": c.get("source", ""), "page": c.get("page", 0), "score": c.get("score", 0.0)} for c in chunks]
