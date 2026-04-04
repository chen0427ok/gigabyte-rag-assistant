"""
Step 3 - Chunking: Convert raw_specs.json into flat text chunks for embedding.

Each chunk = one spec entry for one model, formatted as a natural sentence.
Output includes metadata (model, key) to support filtered retrieval later.

Output format (data/chunks/chunks.json):
[
  {
    "id": 0,
    "model": "AORUS MASTER 16 BZH",
    "key": "顯示晶片",
    "text": "技嘉筆電 AORUS MASTER 16 BZH 的 顯示晶片 規格為：NVIDIA® GeForce RTX™ 5090 Laptop GPU, 24GB GDDR7, ..."
  },
  ...
]
"""

import json
from pathlib import Path

INPUT_JSON = Path("data/raw/raw_specs.json")
OUTPUT_JSON = Path("data/chunks/chunks.json")


def build_chunks(data: dict) -> list[dict]:
    chunks = []
    chunk_id = 0

    for model_name, specs in data.items():
        for key, value in specs.items():
            clean_value = value.replace("\n", ", ")
            text = f"技嘉筆電 {model_name} 的 {key} 規格為：{clean_value}。"
            chunks.append({
                "id": chunk_id,
                "model": model_name,
                "key": key,
                "text": text,
            })
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"找不到 {INPUT_JSON}，請先確認 raw_specs.json 存在。")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    chunks = build_chunks(data)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[chunker] 共產生 {len(chunks)} 個 chunks")
    print(f"[chunker] 儲存至 {OUTPUT_JSON}")
    print("\n--- 前 3 筆預覽 ---")
    for chunk in chunks[:3]:
        print(f"[{chunk['id']}] {chunk['model']} | {chunk['key']}")
        print(f"     {chunk['text'][:80]}...")
        print()
