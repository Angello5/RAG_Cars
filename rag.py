import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker
import torch


BASE = Path(__file__).resolve().parent
ENV_PATH = BASE / ".env"
PROCESSED_DIR = BASE / "data" / "processed"
INDEX_DIR = BASE / "index" / "chroma_autos"

load_dotenv(dotenv_path=ENV_PATH, override=True)

SYSTEM = (
    "Eres un asesor experto en autos para Perú y responde en Español siempre.\n"
    "Responde SOLO con el contexto. Especifica versión/año si aplica.\n"
    "Si no está en el contexto, di claramente que no tienes ese dato.\n"
)

ALIASES = {
    "vw": "volkswagen",
    "ty": "toyota"
}


def make_groq_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(f"Falta GROQ_API_KEY en tu entorno (.env): {ENV_PATH}")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _guess_year(text: str) -> str:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return m.group(1) if m else ""


def _guess_model_from_filename(filename_stem: str, make: str) -> str:

    stem = filename_stem.replace("-", "_")
    tokens = [t for t in stem.split("_") if t]

    stop = {
        "CATALOGO", "CATALOG", "FICHA", "TECNICA", "TÉCNICA",
        "PERU", "PERÚ", "PDF", "AUTO", "AUTOS", "VEHICULO", "VEHÍCULO",
        "NUEVO", "NUEVA", "MODELO", "VERSION", "VERSIÓN"
    }

    make_up = make.upper()

    candidates = []
    for t in tokens:
        up = t.upper()
        if up in stop:
            continue
        if up == make_up:
            continue
        if re.fullmatch(r"19\d{2}|20\d{2}", t):
            continue
        if len(t) < 3:
            continue
        candidates.append(t)

    if candidates:
        return candidates[0].title()

    return filename_stem.title()


def load_md_documents(processed_root: Path):
    """
    Carga .md y añade metadata consistente con tu index.py:
    - make / make_norm desde carpeta
    - model / model_norm inferido desde nombre del archivo
    - year si aparece en el nombre
    - source_rel para citar fácil
    """
    docs = []
    for md in processed_root.rglob("*.md"):
        loaded = TextLoader(str(md), encoding="utf-8").load()

        rel = md.relative_to(processed_root)
        parts = rel.parts
        make = parts[0] if len(parts) >= 1 else "unknown"

        model = _guess_model_from_filename(md.stem, make)
        year = _guess_year(md.stem)

        make_norm = make.strip().lower()
        model_norm = model.strip().lower()

        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["make"] = make
            d.metadata["model"] = model
            if year:
                d.metadata["year"] = year
            d.metadata["make_norm"] = make_norm
            d.metadata["model_norm"] = model_norm
            d.metadata["source_rel"] = str(rel)

        docs.extend(loaded)

    return docs


def format_sources(docs) -> List[str]:
    out = []
    for d in docs:
        # prioriza source_rel si existe
        src_rel = (d.metadata or {}).get("source_rel")
        if src_rel:
            out.append(src_rel)
            continue

        src = (d.metadata or {}).get("source", "")
        if src:
            try:
                out.append(str(Path(src)).replace(str(PROCESSED_DIR) + "/", ""))
            except Exception:
                out.append(src)

    return list(dict.fromkeys(out))


class RAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": self.device}
        )

        # Cargar Chroma persistido
        self.vector_db = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=self.embeddings
        )

        # Builds BM25 con metadata enriquecida (importante para filtrar)
        base_docs = load_md_documents(PROCESSED_DIR)
        base_chunks = self.splitter.split_documents(base_docs)

        self.bm25 = BM25Retriever.from_documents(base_chunks)
        self.bm25.k = 12

        self.reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=(self.device == "cuda")
        )

        self.client = make_groq_client()
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        # Catálogo de makes disponible (automático, no hardcode)
        self.known_makes = {
            p.name.lower() for p in PROCESSED_DIR.iterdir() if p.is_dir()
        }

    def detect_make_model(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        q = query.lower()

        # make por alias
        for alias, real in ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", q):
                make = real.lower()
                break
        else:
            make = None
            for m in self.known_makes:
                if re.search(rf"\b{re.escape(m)}\b", q):
                    make = m
                    break

        # model (MVP): detecta por presencia literal
        # (mejorable: construir diccionario de modelos por marca desde filenames/metadata)
        model = None
        m = re.search(r"\bamarok\b", q)
        if m:
            model = "amarok"
        else:
            m = re.search(r"\bcorolla\b", q)
            if m:
                model = "corolla"

        return make, model

    def retrieve(self, query: str, topk: int = 5):
        make, model = self.detect_make_model(query)

        where = {}
        if make:
            where["make_norm"] = make
        if model:
            where["model_norm"] = model
        
        chroma_where = None
        if len(where) == 1:
            chroma_where = where
        elif len(where) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where.items()]}
        if chroma_where:
            vec_docs = self.vector_db.similarity_search(query, k=12, filter=chroma_where)
        else:
            vec_docs = self.vector_db.similarity_search(query, k=12)

        # 2) BM25 global (luego filtramos por metadata si aplica)
        bm_docs = self.bm25.invoke(query)

        # merge + dedupe
        pool, seen = [], set()
        for d in vec_docs + bm_docs:
            key = (d.page_content[:200], (d.metadata or {}).get("source"))
            if key not in seen:
                pool.append(d)
                seen.add(key)

        # filtro final por metadata (evita “Toyota Corolla” cuando se pidió Amarok)
        if where:
            def ok(d):
                md = d.metadata or {}
                if "make_norm" in where and md.get("make_norm") != where["make_norm"]:
                    return False
                if "model_norm" in where and md.get("model_norm") != where["model_norm"]:
                    return False
                return True

            pool = [d for d in pool if ok(d)]

        # Si no hay contexto válido, devolvemos vacío (y no citamos nada)
        if not pool:
            return []

        # rerank
        pairs = [[query, d.page_content] for d in pool]
        scores = self.reranker.compute_score(pairs, normalize=True)
        order = sorted(range(len(pool)), key=lambda i: -scores[i])

        return [pool[i] for i in order[:topk]]

    def ask(
        self,
        question: str,
        topk: int = 5,
        max_tokens: int = 400,
        temperature: float = 0.3,
        max_chars_per_doc: int = 2500
    ) -> Tuple[str, List[str], object]:

        docs = self.retrieve(question, topk=topk)

        if not docs:
            return "No tengo información sobre eso en el contexto proporcionado.", [], None

        context = "\n\n---\n\n".join(d.page_content[:max_chars_per_doc] for d in docs)

        user = (
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n"
            f"Respuesta:"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        cites = format_sources(docs)
        return answer, cites, usage