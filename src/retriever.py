"""
Step 5 - Retrieval

輸入：使用者問題（字串）
輸出：top-k 個最相關的 chunks（含 model / key / text / score）

流程：
  1. 載入 FAISS index + metadata
  2. 用 bge-m3 將問題 encode 成 query vector
  3. FAISS search → 取得 top-k 的 index 編號與 cosine score
  4. 用 metadata 把編號換回文字，回傳結果
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_FILE = Path("data/index/specs.faiss")
META_FILE  = Path("data/index/metadata.json")
MODEL_NAME = "BAAI/bge-m3"


class Retriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata: list[dict] = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, ids = self.index.search(query_vec, k=self.top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            chunk = self.metadata[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results


if __name__ == "__main__":
    retriever = Retriever(top_k=3)

    test_queries = [
        "BZH 的 GPU 是什麼？",
        "這台筆電的螢幕規格",
        "有幾個 USB 連接埠？",
        "重量是多少",
        "What is the RAM capacity?",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        results = retriever.retrieve(query)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['model']} | {r['key']}")
            print(f"           {r['text'][:80]}...")
