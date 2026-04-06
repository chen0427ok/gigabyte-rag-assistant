"""
Hybrid Retriever - BM25 稀疏搜尋 + FAISS 密集搜尋，RRF 合併排名

解決純 dense search 對型號代碼（BZH/BYH/BXH）無法精確區分的問題。

RRF 公式：score = 1/(k + rank_dense) + 1/(k + rank_sparse)，k=60

特殊處理：
  - Tokenizer：拆分英數混合 token（usb3.2 → usb + 3.2），去除標點
  - Model-aware：偵測 query 中提到的型號，確保每個型號至少 1 個 chunk

使用方式：
    from hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(top_k=3)
    chunks = retriever.retrieve("BZH 和 BYH 的 GPU 有什麼不同？")
"""

import json
import re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_FILE = Path("data/index/specs.faiss")
META_FILE  = Path("data/index/metadata.json")
MODEL_NAME = "BAAI/bge-m3"

# RRF 超參數：k=60 是原始論文建議值
RRF_K = 60
# 第一階段候選數：從 dense/sparse 各取前 N 個再 fuse
CANDIDATE_N = 20

# 可識別的型號代碼（出現在 query 中用來做 model-aware 分配）
MODEL_CODES = ["BZH", "BYH", "BXH"]


def _tokenize(text: str) -> list[str]:
    """
    混合中英文 tokenizer：
    - 先去除標點（逗號、句號、括號等）
    - 英數混合 token 拆分（usb3.2 → usb, 3.2）
    - 中文字逐字拆開
    """
    # 去除標點，保留英數字、中文、空白
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    for part in text.split():
        if not part:
            continue
        if all(ord(c) < 128 for c in part):
            # 純 ASCII：拆分字母與數字邊界（usb3 → usb, 3; gen2 → gen, 2）
            sub_tokens = re.findall(r"[a-zA-Z]+|[0-9]+(?:\.[0-9]+)?", part)
            tokens.extend(t.lower() for t in sub_tokens if t)
        else:
            # 中英混合：英文子串保留，中文逐字
            buf = ""
            for c in part:
                if ord(c) < 128:
                    buf += c
                else:
                    if buf:
                        sub = re.findall(r"[a-zA-Z]+|[0-9]+(?:\.[0-9]+)?", buf)
                        tokens.extend(t.lower() for t in sub if t)
                        buf = ""
                    tokens.append(c)
            if buf:
                sub = re.findall(r"[a-zA-Z]+|[0-9]+(?:\.[0-9]+)?", buf)
                tokens.extend(t.lower() for t in sub if t)
    return tokens


def _detect_models(query: str) -> list[str]:
    """偵測 query 中提到的型號代碼，回傳如 ['BZH', 'BYH']"""
    return [code for code in MODEL_CODES if code in query.upper()]


class HybridRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        # Dense: FAISS + bge-m3
        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata: list[dict] = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.embed_model = SentenceTransformer(MODEL_NAME)

        # Sparse: BM25 on tokenized corpus
        corpus_tokens = [_tokenize(chunk["text"]) for chunk in self.metadata]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str) -> list[dict]:
        n = min(CANDIDATE_N, len(self.metadata))

        # ── Dense retrieval ──────────────────────────────────────────────────
        query_vec = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        dense_scores, dense_ids = self.index.search(query_vec, k=n)
        # dense_rank[chunk_id] = 1-based rank
        dense_rank: dict[int, int] = {
            int(idx): rank + 1
            for rank, idx in enumerate(dense_ids[0])
        }

        # ── Sparse retrieval (BM25) ──────────────────────────────────────────
        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        # top-N sparse indices by descending BM25 score
        sparse_ids = np.argsort(bm25_scores)[::-1][:n]
        # sparse_rank[chunk_id] = 1-based rank
        sparse_rank: dict[int, int] = {
            int(idx): rank + 1
            for rank, idx in enumerate(sparse_ids)
        }

        # ── RRF fusion ───────────────────────────────────────────────────────
        candidate_ids = set(dense_rank) | set(sparse_rank)
        rrf_scores: dict[int, float] = {}
        for cid in candidate_ids:
            dr = dense_rank.get(cid, n + 1)   # 未出現 → 排在末尾
            sr = sparse_rank.get(cid, n + 1)
            rrf_scores[cid] = 1 / (RRF_K + dr) + 1 / (RRF_K + sr)

        # 全部候選按 RRF 排序
        sorted_candidates = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── Model-aware 分配 ─────────────────────────────────────────────────
        mentioned_models = _detect_models(query)

        if len(mentioned_models) >= 2:
            # 比較類 query：確保每個提到的型號至少有 1 個 chunk
            selected: list[int] = []
            used: set[int] = set()

            # 每個型號挑 RRF 分最高的 chunk
            for model_code in mentioned_models:
                for cid in sorted_candidates:
                    if cid in used:
                        continue
                    if model_code in self.metadata[cid]["model"]:
                        selected.append(cid)
                        used.add(cid)
                        break

            # 剩餘 slot 用全域 RRF 排名填滿
            for cid in sorted_candidates:
                if len(selected) >= self.top_k:
                    break
                if cid not in used:
                    selected.append(cid)
                    used.add(cid)
        else:
            # 非比較類 query：直接取 top_k
            selected = sorted_candidates[: self.top_k]

        # ── 組裝結果 ─────────────────────────────────────────────────────────
        results = []
        for cid in selected[: self.top_k]:
            chunk = self.metadata[cid].copy()
            chunk["score"] = round(rrf_scores[cid], 6)
            chunk["dense_rank"] = dense_rank.get(cid, None)
            chunk["sparse_rank"] = sparse_rank.get(cid, None)
            results.append(chunk)

        return results


if __name__ == "__main__":
    retriever = HybridRetriever(top_k=3)

    test_queries = [
        "BZH 和 BYH 的 GPU 有什麼不同？",   # Q13 dense 失敗案例
        "BZH 的顯示晶片是什麼？",
        "How many USB ports does this laptop have?",
        "What is the RAM capacity?",
        "BZH 和 BXH 的 GPU 有什麼不同？",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        results = retriever.retrieve(query)
        for r in results:
            print(
                f"  [rrf={r['score']:.6f}] "
                f"dense_rank={r['dense_rank']} sparse_rank={r['sparse_rank']} | "
                f"{r['model']} | {r['key']}"
            )
