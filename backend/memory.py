"""Vector memory of de-identified session summaries (ChromaDB, local embeddings)."""
from pathlib import Path

import chromadb

PATH = Path(__file__).resolve().parent.parent / "data" / "memory"
# Cosine distance cutoff. Probed with MiniLM: related notes scored 0.52-0.65, unrelated 0.80+.
MAX_DISTANCE = 0.70

_col = None


def _collection():
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path=str(PATH))
        _col = client.get_or_create_collection("summaries", metadata={"hnsw:space": "cosine"})
    return _col


def add(session_id, summary, date):
    _collection().upsert(ids=[str(session_id)], documents=[summary],
                         metadatas=[{"session_id": session_id, "date": date}])


def search(text, exclude_session, k=2):
    col = _collection()
    if col.count() == 0:
        return []
    r = col.query(query_texts=[text], n_results=min(k, col.count()),
                  where={"session_id": {"$ne": exclude_session}})
    return [{"summary": doc, "date": meta["date"], "distance": round(dist, 3)}
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
            if dist <= MAX_DISTANCE]
