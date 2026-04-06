"""
Step 8 - System Evaluation

定量指標：每題的 TTFT、TPS、token 數
定性分析：Retrieval 是否命中、答案是否正確

使用方式：
    uv run python src/evaluate.py
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

from retriever import Retriever
from hybrid_retriever import HybridRetriever
from generator import Generator, DEFAULT_MODEL_PATH, MODEL_7B_PATH

console = Console()

# ── 評測題組 ────────────────────────────────────────────────────────────────
# expected_key: retriever 應該要找到的規格欄位
# expected_hint: 答案中應該要出現的關鍵字（用來判斷回答正確性）
BENCHMARK = [
    {
        "id": 1,
        "category": "直接查詢",
        "query": "BZH 的顯示晶片是什麼？",
        "expected_key": "顯示晶片",
        "expected_hint": ["5090", "RTX"],
    },
    {
        "id": 2,
        "category": "直接查詢",
        "query": "這台筆電的記憶體最大是多少？",
        "expected_key": "記憶體",
        "expected_hint": ["64GB"],
    },
    {
        "id": 3,
        "category": "直接查詢",
        "query": "螢幕解析度是多少？",
        "expected_key": "顯示器",
        "expected_hint": ["2560", "1600"],
    },
    {
        "id": 4,
        "category": "直接查詢",
        "query": "電池容量是多少？",
        "expected_key": "電池",
        "expected_hint": ["99Wh", "99"],
    },
    {
        "id": 5,
        "category": "直接查詢",
        "query": "鍵盤有幾個 RGB 燈光區域？",
        "expected_key": "鍵盤種類",
        "expected_hint": ["3"],   # 接受 "3區" 或 "三個" 等，只核對數字
    },
    {
        "id": 6,
        "category": "型號比較",
        "query": "BZH 和 BXH 的 GPU 有什麼不同？",
        "expected_key": "顯示晶片",
        "expected_hint": ["5090", "5070"],
    },
    {
        "id": 7,
        "category": "型號比較",
        "query": "哪個型號的顯示晶片最強？",
        "expected_key": "顯示晶片",
        "expected_hint": ["BZH", "5090"],
    },
    {
        "id": 8,
        "category": "是非推論",
        "query": "這台筆電有支援 Thunderbolt 5 嗎？",
        "expected_key": "連接埠",
        "expected_hint": ["是", "支援"],
    },
    {
        "id": 9,
        "category": "是非推論",
        "query": "螢幕有支援 Dolby Vision 嗎？",
        "expected_key": "顯示器",
        "expected_hint": ["Dolby"],  # 7B Q3_K_M 已知在長清單末尾會漏判
    },
    {
        "id": 10,
        "category": "英文查詢",
        "query": "What is the RAM capacity?",
        "expected_key": "記憶體",
        "expected_hint": ["64GB"],
    },
    {
        "id": 11,
        "category": "英文查詢",
        "query": "How many USB ports does this laptop have?",
        "expected_key": "連接埠",
        "expected_hint": ["USB", "Type-A", "Type-C"],
    },
    {
        "id": 12,
        "category": "英文查詢",
        "query": "What GPU does the BXH model have?",
        "expected_key": "顯示晶片",
        "expected_hint": ["5070", "RTX"],
    },
    {
        "id": 13,
        "category": "型號比較",
        "query": "BZH 和 BYH 的 GPU 有什麼不同？",
        "expected_key": "顯示晶片",
        "expected_hint": ["5090", "5080"],  # dense search 已知會漏掉 BZH，hybrid 應能修正
    },
]


def check_retrieval(chunks: list[dict], expected_key: str) -> bool:
    """Top-3 chunks 裡有沒有命中 expected_key。"""
    return any(c["key"] == expected_key for c in chunks)


def check_answer(answer: str, hints: list[str]) -> bool:
    """答案裡有沒有包含所有 hint 關鍵字（大小寫不敏感）。"""
    answer_lower = answer.lower()
    return all(h.lower() in answer_lower for h in hints)


def run_benchmark(top_k: int = 3, model: str = "3b", hybrid: bool = False) -> None:
    model_path = MODEL_7B_PATH if model == "7b" else DEFAULT_MODEL_PATH
    retriever_label = "hybrid" if hybrid else "dense"
    console.print(f"\n[bold cyan]載入 RAG Pipeline（model={model}, retriever={retriever_label}）...[/bold cyan]")
    retriever = HybridRetriever(top_k=top_k) if hybrid else Retriever(top_k=top_k)
    generator = Generator(model_path=model_path)
    console.print("[bold green]開始評測！[/bold green]\n")

    results = []

    for item in BENCHMARK:
        console.rule(f"[bold]#{item['id']} [{item['category']}] {item['query']}[/bold]")

        # Retrieve
        chunks = retriever.retrieve(item["query"])
        retrieval_hit = check_retrieval(chunks, item["expected_key"])

        top1 = chunks[0]
        console.print(
            f"  Top-1: [cyan]{top1['model']}[/cyan] | [magenta]{top1['key']}[/magenta] "
            f"(score={top1['score']})  "
            + ("[green]✓ Hit[/green]" if retrieval_hit else "[red]✗ Miss[/red]")
        )

        # Generate
        console.print("  回答：", end="")
        result = generator.generate(item["query"], chunks)
        answer_correct = check_answer(result["answer"], item["expected_hint"])
        m = result["metrics"]

        console.print(
            f"  [dim]TTFT={m['ttft_sec']}s | TPS={m['tps']} | tokens={m['total_tokens']}[/dim]  "
            + ("[green]✓ Correct[/green]" if answer_correct else "[red]✗ Incorrect[/red]")
        )

        results.append({
            **item,
            "retrieval_hit": retrieval_hit,
            "answer_correct": answer_correct,
            "answer": result["answer"],
            "ttft_sec": m["ttft_sec"],
            "tps": m["tps"],
            "total_tokens": m["total_tokens"],
            "total_sec": m["total_sec"],
            "mem_after_gen_gb": m.get("mem_after_gen_gb"),
            "retrieved_chunks": chunks,
        })

    _print_report(results)
    _save_results(results, model, hybrid)


def _print_report(results: list[dict]) -> None:
    console.rule("\n[bold]📊 Benchmark Report[/bold]")

    # ── 定量指標表 ──
    table = Table(box=box.MARKDOWN, show_header=True, header_style="bold cyan")
    table.add_column("#",        justify="right",  width=3)
    table.add_column("Category", width=10)
    table.add_column("Query",    width=28)
    table.add_column("Ret.",     justify="center", width=5)
    table.add_column("Ans.",     justify="center", width=5)
    table.add_column("TTFT(s)",  justify="right",  width=7)
    table.add_column("TPS",      justify="right",  width=6)
    table.add_column("Tokens",   justify="right",  width=7)

    ttft_list, tps_list = [], []

    for r in results:
        ret_mark = "✓" if r["retrieval_hit"] else "✗"
        ans_mark = "✓" if r["answer_correct"] else "✗"
        ret_style = "green" if r["retrieval_hit"] else "red"
        ans_style = "green" if r["answer_correct"] else "red"

        table.add_row(
            str(r["id"]),
            r["category"],
            r["query"][:26] + ("…" if len(r["query"]) > 26 else ""),
            f"[{ret_style}]{ret_mark}[/{ret_style}]",
            f"[{ans_style}]{ans_mark}[/{ans_style}]",
            str(r["ttft_sec"]) if r["ttft_sec"] else "-",
            str(r["tps"]),
            str(r["total_tokens"]),
        )
        if r["ttft_sec"]:
            ttft_list.append(r["ttft_sec"])
        tps_list.append(r["tps"])

    console.print(table)

    # ── 彙總統計 ──
    n = len(results)
    ret_acc = sum(r["retrieval_hit"] for r in results) / n * 100
    ans_acc = sum(r["answer_correct"] for r in results) / n * 100
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tps  = sum(tps_list) / len(tps_list)

    console.print(f"\n[bold]彙總統計（共 {n} 題）[/bold]")
    console.print(f"  Retrieval Hit Rate : [cyan]{ret_acc:.1f}%[/cyan]  ({sum(r['retrieval_hit'] for r in results)}/{n})")
    console.print(f"  Answer Accuracy    : [cyan]{ans_acc:.1f}%[/cyan]  ({sum(r['answer_correct'] for r in results)}/{n})")
    console.print(f"  Avg TTFT           : [cyan]{avg_ttft:.3f}s[/cyan]")
    console.print(f"  Avg TPS            : [cyan]{avg_tps:.2f}[/cyan]")

    # ── 定性分析 ──
    console.rule("\n[bold]📝 定性分析[/bold]")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"hit": 0, "correct": 0, "total": 0}
        categories[cat]["total"] += 1
        categories[cat]["hit"] += r["retrieval_hit"]
        categories[cat]["correct"] += r["answer_correct"]

    for cat, stats in categories.items():
        console.print(
            f"  [{cat}]  "
            f"Retrieval {stats['hit']}/{stats['total']}  "
            f"Answer {stats['correct']}/{stats['total']}"
        )


def _save_results(results: list[dict], model: str, hybrid: bool = False) -> None:
    retriever = "hybrid" if hybrid else "semantic"
    out = Path(f"data/eval_results_{model}_{retriever}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[dim]結果已儲存至 {out}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["3b", "7b"], default="3b",
                        help="選擇模型：3b（預設）或 7b")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hybrid", action="store_true",
                        help="使用 Hybrid Retriever（BM25 + FAISS + RRF）")
    args = parser.parse_args()
    run_benchmark(top_k=args.top_k, model=args.model, hybrid=args.hybrid)
