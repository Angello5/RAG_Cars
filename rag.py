import os
from pathlib import Path
from typing import List, Tuple

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

def load_md_documents(processed_root: Path):
    docs = []
    for md in processed_root.rglob("*.md"):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    return docs

def format_sources(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source", "")
        if src:
            try:
                src_rel = str(Path(src)).replace(str(PROCESSED_DIR) + "/", "")
            except Exception:
                src_rel = src
            out.append(src_rel)
    return list(dict.fromkeys(out))

def make_groq_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Falta GROQ_API_KEY en tu entorno (.env).")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

class RAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": self.device}
        )

        # Cargar Chroma pers
        # Nota: según versión, el parámetro puede ser embedding_function o embedding.
        # Este es el más común en langchain-community:
        self.vector_db = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=self.embeddings
        )

        self.vec_retriever = self.vector_db.as_retriever(search_kwargs={"k": 12})

        base_docs = load_md_documents(PROCESSED_DIR)
        base_chunks = self.splitter.split_documents(base_docs)

        self.bm25 = BM25Retriever.from_documents(base_chunks)
        self.bm25.k = 12

        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=(self.device == "cuda"))

        self.client = make_groq_client()
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def retrieve(self, query: str, topk: int = 5):
        vec_docs = self.vec_retriever.invoke(query)
        bm_docs = self.bm25.invoke(query)

        pool, seen = [], set()
        for d in vec_docs + bm_docs:
            key = (d.page_content[:200], d.metadata.get("source"))
            if key not in seen:
                pool.append(d)
                seen.add(key)

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