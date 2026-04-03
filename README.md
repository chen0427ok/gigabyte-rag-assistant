# GIGABYTE AORUS MASTER 16 AM6H RAG Assistant

純 Python 實作的 RAG 問答系統，針對 GIGABYTE AORUS MASTER 16 AM6H 產品規格。

## 技術棧
- **推論模型**：Qwen2.5-3B-Instruct Q4_K_M（via llama-cpp-python + Metal）
- **Embedding**：BAAI/bge-m3（sentence-transformers）
- **向量搜尋**：faiss-cpu + numpy（純手寫 RAG，無 LangChain/LlamaIndex）
- **環境管理**：uv
- **語系支援**：繁體中文 + 英文混合

## 快速開始
```bash
uv sync
```
