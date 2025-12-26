import os, json
from pathlib import Path
from typing import List
from pydantic import BaseModel

from docling.document_converter import DocumentConverter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch


BASE = Path(__file__).resolve().parent
PDF_DIR = BASE
PROCESSED_DIR = BASE / "data" / "processed"
INDEX_DIR = BASE / "index" / "chroma_autos"

for p in [PROCESSED_DIR, INDEX_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SKIP_DIR_NAMES = {"data", "index", ".git", ".ipynb_checkpoints", ".idea", ".venv", "env", "venv"}

def should_skip(path: Path) -> bool:
    return any(part.startswith(".") or part in SKIP_DIR_NAMES for part in path.parts)

class ParseResult(BaseModel):
    pdf_path: str
    md_path: str
    json_meta_path: str
    chars: int
    used: str  # docling

def rel_to_base(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except Exception:
        return Path(os.path.relpath(path, base))

def convert_pdf_docling(pdf_path: Path):
    conv = DocumentConverter()
    res = conv.convert(str(pdf_path))
    md_text = res.document.export_to_markdown()
    meta = res.document.as_dict() if hasattr(res.document, "as_dict") else {"note": "no-as_dict"}
    return md_text, meta

def convert_pdf(pdf_path: Path, out_md: Path, out_json: Path):
    md_text, meta = convert_pdf_docling(pdf_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return ParseResult(
        pdf_path=str(pdf_path),
        md_path=str(out_md),
        json_meta_path=str(out_json),
        chars=len(md_text),
        used="docling"
    )

def ingest_all(pdf_list: List[Path], processed_root: Path) -> List[ParseResult]:
    results = []
    for pdf in sorted(pdf_list):
        if not pdf.exists():
            print(f"[WARN] No existe: {pdf}")
            continue
        rel = rel_to_base(pdf, PDF_DIR)
        out_md = processed_root / rel.with_suffix(".md")
        out_json = processed_root / rel.with_suffix(".json")
        r = convert_pdf(pdf, out_md, out_json)
        print(f"✓ docling {rel} -> {out_md} ({r.chars} chars)")
        results.append(r)
    return results

def load_md_documents(processed_root: Path):
    docs = []
    for md in processed_root.rglob("*.md"):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    return docs

def main():
    # Descubrir PDFs
    candidates = []
    for p in PDF_DIR.rglob("*.pdf"):
        if should_skip(p):
            continue
        if PROCESSED_DIR in p.parents:
            continue
        candidates.append(p.resolve())

    print("PDFs detectados:", len(candidates))
    ingest_all(candidates, PROCESSED_DIR)

    # Indexar
    docs = load_md_documents(PROCESSED_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=130)
    chunks = splitter.split_documents(docs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": device}
    )

    # Crea (o recrea) índice
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR)
    )

    # Algunas instalaciones/versiones persisten automáticamente; si tu versión requiere persist explícito, descomenta:
    # vector_db.persist()

    print(f"Index listo. Chunks: {len(chunks)} | Persist: {INDEX_DIR}")

if __name__ == "__main__":
    main()