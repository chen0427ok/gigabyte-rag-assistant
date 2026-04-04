"""
Step 4 - Embedding + Vector Index

使用 BAAI/bge-m3 將 51 個 chunks 向量化，建立 FAISS index。

- 向量正規化後使用 IndexFlatIP（內積 = cosine similarity）
- 輸出：
    data/index/specs.faiss   ← FAISS binary index
    data/index/metadata.json ← chunk 的 id / model / key / text 對照表
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_JSON = Path("data/chunks/chunks.json")
INDEX_DIR   = Path("data/index")
INDEX_FILE  = INDEX_DIR / "specs.faiss"
META_FILE   = INDEX_DIR / "metadata.json"

MODEL_NAME = "BAAI/bge-m3"


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    texts = [c["text"] for c in chunks]

    print(f"[embedder] Encoding {len(texts)} chunks with {MODEL_NAME} ...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 正規化，讓 IP = cosine
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    print(f"[embedder] Embedding dim: {dim}")

    index = faiss.IndexFlatIP(dim)   # Inner Product on normalized vecs = cosine
    index.add(embeddings.astype(np.float32))
    print(f"[embedder] FAISS index built: {index.ntotal} vectors")
    return index


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_JSON)

    model = SentenceTransformer(MODEL_NAME)

    index = build_index(chunks, model)

    # 儲存 index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"[embedder] FAISS index saved → {INDEX_FILE}")

    # 儲存 metadata（去掉 text 以外的欄位也保留，方便 retriever 用）
    META_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[embedder] Metadata saved → {META_FILE}")

    print("\n--- 快速驗證：以 '顯示晶片' 做相似度查詢 ---")
    query = "顯示晶片"
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(q_vec.astype(np.float32), k=3)
    for score, idx in zip(scores[0], ids[0]):
        print(f"  score={score:.4f} | {chunks[idx]['model']} | {chunks[idx]['key']}")
        print(f"           {chunks[idx]['text'][:80]}...")
