"""
Step 6 - Generation with Streaming

流程：
  1. 載入 Qwen2.5-3B-Instruct Q4_K_M（llama-cpp-python，Metal 加速）
  2. 接收 retrieved chunks，組成 prompt
  3. Streaming 逐 token 輸出回答
  4. 同步量測 TTFT（首字延遲）與 TPS（生成速度）
"""

import time
from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")

SYSTEM_PROMPT = """你是一個專業的技嘉筆電產品規格小幫手。
請根據以下提供的規格資料，用繁體中文回答使用者的問題。
回答要精確、簡潔，直接引用規格數據。
如果規格資料中找不到相關資訊，請直接說「資料中未找到相關規格」。"""


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n".join(
        f"- {c['text']}" for c in chunks
    )
    user_message = f"""參考規格資料：
{context}

使用者問題：{query}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


class Generator:
    def __init__(self):
        print(f"[generator] 載入模型：{MODEL_PATH}")
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,   # 全部 layer 交給 Metal GPU
            n_ctx=2048,
            verbose=False,
        )
        print("[generator] 模型載入完成")

    def generate(self, query: str, chunks: list[dict]) -> dict:
        messages = build_prompt(query, chunks)

        t_start = time.perf_counter()
        first_token_time = None
        full_text = ""
        token_count = 0

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,      # 規格問答不需要創意，低溫保持精確
            repeat_penalty=1.1,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if not token:
                continue

            if first_token_time is None:
                first_token_time = time.perf_counter()

            print(token, end="", flush=True)
            full_text += token
            token_count += 1

        t_end = time.perf_counter()
        print()  # 換行

        ttft = (first_token_time - t_start) if first_token_time else None
        total_time = t_end - t_start
        tps = token_count / total_time if total_time > 0 else 0

        return {
            "answer": full_text,
            "metrics": {
                "ttft_sec":    round(ttft, 3) if ttft else None,
                "tps":         round(tps, 2),
                "total_tokens": token_count,
                "total_sec":   round(total_time, 3),
            },
        }


if __name__ == "__main__":
    from retriever import Retriever

    retriever = Retriever(top_k=3)
    generator = Generator()

    test_queries = [
        "BZH 的顯示晶片是什麼規格？",
        "這台筆電的螢幕有支援 HDR 嗎？",
        "有幾個 Thunderbolt 連接埠？",
        "What is the battery capacity?",
    ]

    for query in test_queries:
        print(f"\n{'═'*60}")
        print(f"問題：{query}")
        print("回答：", end="")

        chunks = retriever.retrieve(query)
        result = generator.generate(query, chunks)

        print(f"\n📊 TTFT={result['metrics']['ttft_sec']}s  "
              f"TPS={result['metrics']['tps']}  "
              f"tokens={result['metrics']['total_tokens']}  "
              f"total={result['metrics']['total_sec']}s")
