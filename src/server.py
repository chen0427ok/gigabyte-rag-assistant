"""
FastAPI + Gradio ChatUI Server

架構：
  - FastAPI 負責 REST API（/api/chat SSE、/api/health）
  - Gradio ChatInterface 掛載於根路徑 /，提供網頁聊天介面
  - RAG Pipeline（Retriever + Generator）在啟動時載入，所有請求共用同一實例

使用方式：
    uv run python src/server.py
    uv run python src/server.py --hybrid       # 使用 Hybrid Retriever
    uv run python src/server.py --model 7b     # 使用 7B 模型
    uv run python src/server.py --port 7860

啟動後瀏覽：
    Chat UI  → http://localhost:7860
    API docs → http://localhost:7860/docs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Generator as GenType

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# 讓 src/ 下的模組可以直接 import
sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from hybrid_retriever import HybridRetriever
from generator import Generator, DEFAULT_MODEL_PATH, MODEL_7B_PATH

# ── 全域 RAG Pipeline（啟動時初始化）────────────────────────────────────────
retriever: Retriever | HybridRetriever | None = None
generator: Generator | None = None


def _init_pipeline(model: str = "3b", hybrid: bool = False, top_k: int = 3) -> None:
    global retriever, generator
    model_path = MODEL_7B_PATH if model == "7b" else DEFAULT_MODEL_PATH
    retriever = HybridRetriever(top_k=top_k) if hybrid else Retriever(top_k=top_k)
    generator = Generator(model_path=model_path)


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="GIGABYTE RAG Assistant", version="1.0.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "retriever": type(retriever).__name__}


@app.get("/api/chat")
def api_chat(query: str, top_k: int = 3) -> StreamingResponse:
    """
    SSE 串流回答端點。

    回傳格式（text/event-stream）：
        data: {"chunk": "部分回答..."}\n\n
        data: {"chunk": "..."}\n\n
        data: [DONE]\n\n
    """
    def _stream() -> GenType[str, None, None]:
        chunks = retriever.retrieve(query)
        messages = _build_messages(query, chunks)

        stream = generator.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            repeat_penalty=1.1,
            stream=True,
        )

        for event in stream:
            token = event["choices"][0]["delta"].get("content", "")
            if token:
                yield f"data: {json.dumps({'chunk': token}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    from generator import build_prompt
    return build_prompt(query, chunks)


# ── Gradio ChatInterface ─────────────────────────────────────────────────────

def _chat_fn(message: str, history: list) -> GenType[str, None, None]:
    """
    Gradio ChatInterface 的串流回答函式。
    使用 yield 逐步返回累積回答，Gradio 自動處理串流顯示。
    """
    chunks = retriever.retrieve(message)

    # 顯示 Retrieved Chunks（附加在回答前，折疊顯示）
    chunk_info = "\n".join(
        f"`[{i+1}]` **{c['model']}** | {c['key']} (score={c['score']})"
        for i, c in enumerate(chunks)
    )

    messages = _build_messages(message, chunks)
    stream = generator.llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
        repeat_penalty=1.1,
        stream=True,
    )

    answer = ""
    for event in stream:
        token = event["choices"][0]["delta"].get("content", "")
        if token:
            answer += token
            yield answer

    # 回答完成後附上 Retrieved Chunks 資訊
    yield f"{answer}\n\n---\n**Retrieved Chunks：**\n{chunk_info}"


def _build_gradio_app() -> gr.Blocks:
    retriever_name = type(retriever).__name__
    model_name = Path(generator.llm.model_path).stem

    with gr.Blocks(title="GIGABYTE RAG Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
# GIGABYTE AORUS MASTER 16 RAG 小幫手
**模型：** `{model_name}` ｜ **Retriever：** `{retriever_name}`

輸入關於 GIGABYTE AORUS MASTER 16 AM6H 系列筆電的問題，系統將從規格資料庫中檢索並回答。
            """
        )

        gr.ChatInterface(
            fn=_chat_fn,
            examples=[
                "BZH 的顯示晶片是什麼？",
                "BZH 和 BYH 的 GPU 有什麼不同？",
                "螢幕解析度是多少？",
                "How many USB ports does this laptop have?",
                "電池容量是多少？",
                "哪個型號的顯示晶片最強？",
            ],
            cache_examples=False,
        )

    return demo


# ── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GIGABYTE RAG FastAPI + Gradio Server")
    parser.add_argument("--model",  choices=["3b", "7b"], default="3b")
    parser.add_argument("--hybrid", action="store_true", help="使用 Hybrid Retriever")
    parser.add_argument("--top-k",  type=int, default=3)
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=7860)
    args = parser.parse_args()

    print(f"[server] 載入 RAG Pipeline（model={args.model}, hybrid={args.hybrid}）...")
    _init_pipeline(model=args.model, hybrid=args.hybrid, top_k=args.top_k)
    print("[server] Pipeline 就緒！")

    # 將 Gradio 掛載到 FastAPI
    demo = _build_gradio_app()
    gr.mount_gradio_app(app, demo, path="/")

    print(f"\n[server] Chat UI  → http://{args.host}:{args.port}/")
    print(f"[server] API docs → http://{args.host}:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
