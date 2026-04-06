"""
feature/llm-judge - Evaluate with Claude as judge

與 evaluate.py 的差別：
  - 用 Claude API 取代關鍵字比對來判斷答案是否正確
  - Ground truth = raw_specs.json 完整規格資料（權威來源）
  - 支援兩種評分模式（--judge-mode 參數）：
      binary（預設）: correct / incorrect + 一句理由
      score         : 三維度 1-5 分（faithfulness / correctness / conciseness）
                      pass 條件：correctness >= 4

使用方式：
    uv run python src/evaluate_judge.py
    uv run python src/evaluate_judge.py --judge-mode score
    uv run python src/evaluate_judge.py --model 7b --judge-mode score
    uv run python src/evaluate_judge.py --hybrid                        # Hybrid Retriever
    uv run python src/evaluate_judge.py --hybrid --judge-mode score
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

from retriever import Retriever
from hybrid_retriever import HybridRetriever
from generator import Generator, DEFAULT_MODEL_PATH, MODEL_7B_PATH
from evaluate import BENCHMARK, check_retrieval

load_dotenv()

console = Console()

JUDGE_MODEL    = "claude-haiku-4-5-20251001"
RAW_SPECS_PATH = Path("data/raw/raw_specs.json")
PASS_THRESHOLD = 4  # score 模式用

# ── Judge prompts ──────────────────────────────────────────────────────────────

BINARY_SYSTEM = """你是一個嚴格的 RAG 系統評審。
你會收到：使用者問題、retrieved context、完整官方規格資料（ground truth）、RAG 系統回答。
請判斷 RAG 系統的回答是否正確回應了使用者的問題。對照 ground truth 判斷，若回答說「找不到相關規格」但 ground truth 中明確有此資料，應判為 incorrect。

請只回應 JSON，不要輸出任何其他文字：
{"correct": true or false, "reason": "<一句話說明>"}"""

SCORE_SYSTEM = """你是一個嚴格的 RAG 系統評審。
你會收到：使用者問題、retrieved context、完整官方規格資料（ground truth）、RAG 系統回答。
請從三個維度各給 1-5 分，並用一句話說明理由。

評分維度：
- faithfulness（忠實度）：答案是否完全基於 retrieved context，沒有捏造 context 中沒有的資訊。
  5=完全基於 context，無幻覺；1=嚴重幻覺
- correctness（正確性）：對照 ground truth，答案是否正確且完整。若回答「找不到」但 ground truth 有此資料，給 1 分。
  5=完全正確且完整；3=部分正確；1=完全錯誤
- conciseness（精簡度）：答案是否簡潔，沒有不必要的冗餘。
  5=非常精簡；3=尚可；1=極度冗長

請嚴格以此 JSON 格式回應，不要輸出任何其他文字：
{
  "faithfulness": {"score": <1-5>, "reason": "<一句話>"},
  "correctness":  {"score": <1-5>, "reason": "<一句話>"},
  "conciseness":  {"score": <1-5>, "reason": "<一句話>"}
}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> str:
    specs = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for model_name, fields in specs.items():
        lines.append(f"=== {model_name} ===")
        for key, value in fields.items():
            lines.append(f"{key}：{value}")
        lines.append("")
    return "\n".join(lines)


def _strip_code_block(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _build_prompt(query: str, ground_truth: str, chunks: list[dict], answer: str) -> str:
    context = "\n".join(f"- {c['text']}" for c in chunks)
    return f"""使用者問題：{query}

Retrieved Context：
{context}

完整官方規格資料（Ground Truth）：
{ground_truth}

RAG 系統回答：
{answer}"""


def judge_binary(
    client: anthropic.Anthropic,
    query: str,
    ground_truth: str,
    chunks: list[dict],
    answer: str,
) -> tuple[bool, str]:
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=128,
        system=BINARY_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(query, ground_truth, chunks, answer)}],
    )
    raw = _strip_code_block(message.content[0].text.strip())
    parsed = json.loads(raw)
    return bool(parsed["correct"]), parsed.get("reason", "")


def judge_score(
    client: anthropic.Anthropic,
    query: str,
    ground_truth: str,
    chunks: list[dict],
    answer: str,
) -> dict:
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=SCORE_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(query, ground_truth, chunks, answer)}],
    )
    raw = _strip_code_block(message.content[0].text.strip())
    return json.loads(raw)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_benchmark_with_judge(
    top_k: int = 3, model: str = "3b", judge_mode: str = "binary", hybrid: bool = False
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "Your claude api key":
        console.print("[bold red]錯誤：請在 .env 設定有效的 ANTHROPIC_API_KEY[/bold red]")
        return

    client = anthropic.Anthropic(api_key=api_key)
    ground_truth = load_ground_truth(RAW_SPECS_PATH)

    model_path = MODEL_7B_PATH if model == "7b" else DEFAULT_MODEL_PATH
    retriever_label = "hybrid" if hybrid else "dense"
    console.print(f"\n[bold cyan]載入 RAG Pipeline（model={model}, retriever={retriever_label}, judge={judge_mode}）...[/bold cyan]")
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
        m = result["metrics"]
        console.print(
            f"  [dim]TTFT={m['ttft_sec']}s | TPS={m['tps']} | tokens={m['total_tokens']}[/dim]"
        )

        # Judge
        console.print("  [dim]Claude 評分中...[/dim]", end=" ")
        answer_pass = False
        scores = {}
        reason = ""

        try:
            if judge_mode == "binary":
                answer_pass, reason = judge_binary(client, item["query"], ground_truth, chunks, result["answer"])
                console.print(
                    ("[green]✓ Correct[/green]" if answer_pass else "[red]✗ Incorrect[/red]")
                    + f"  [dim]{reason}[/dim]"
                )
            else:  # score
                scores = judge_score(client, item["query"], ground_truth, chunks, result["answer"])
                f_s = scores["faithfulness"]["score"]
                c_s = scores["correctness"]["score"]
                p_s = scores["conciseness"]["score"]
                answer_pass = c_s >= PASS_THRESHOLD
                reason = scores["correctness"]["reason"]

                def _col(s: int) -> str:
                    return "green" if s >= 4 else ("yellow" if s == 3 else "red")

                console.print(
                    f"Faith=[{_col(f_s)}]{f_s}[/{_col(f_s)}] "
                    f"Corr=[{_col(c_s)}]{c_s}[/{_col(c_s)}] "
                    f"Conc=[{_col(p_s)}]{p_s}[/{_col(p_s)}]  "
                    + ("[green]✓ Pass[/green]" if answer_pass else "[red]✗ Fail[/red]")
                )
                console.print(f"  [dim]→ {reason}[/dim]")

        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]評分解析失敗：{e}，預設 False[/yellow]")

        time.sleep(0.3)

        results.append({
            **item,
            "retrieval_hit": retrieval_hit,
            "answer_pass": answer_pass,
            "judge_reason": reason,
            "scores": scores,          # score 模式才有內容，binary 模式為 {}
            "answer": result["answer"],
            "ttft_sec": m["ttft_sec"],
            "tps": m["tps"],
            "total_tokens": m["total_tokens"],
            "total_sec": m["total_sec"],
            "mem_after_gen_gb": m.get("mem_after_gen_gb"),
            "retrieved_chunks": chunks,
        })

    _print_report(results, judge_mode)
    _save_results(results, model, judge_mode, hybrid)


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(results: list[dict], judge_mode: str) -> None:
    console.rule(f"\n[bold]📊 Benchmark Report (LLM Judge / {judge_mode})[/bold]")

    table = Table(box=box.MARKDOWN, show_header=True, header_style="bold cyan")
    table.add_column("#",        justify="right",  width=3)
    table.add_column("Category", width=10)
    table.add_column("Query",    width=24)
    table.add_column("Ret.",     justify="center", width=5)

    if judge_mode == "score":
        table.add_column("Faith.", justify="center", width=7)
        table.add_column("Corr.",  justify="center", width=6)
        table.add_column("Conc.",  justify="center", width=6)

    table.add_column("Pass" if judge_mode == "score" else "Correct", justify="center", width=8)
    table.add_column("TTFT(s)", justify="right", width=7)
    table.add_column("TPS",     justify="right", width=6)

    ttft_list, tps_list = [], []
    dim_totals: dict[str, list[int]] = {"faithfulness": [], "correctness": [], "conciseness": []}

    def _styled(s) -> str:
        if not isinstance(s, int):
            return str(s)
        color = "green" if s >= 4 else ("yellow" if s == 3 else "red")
        return f"[{color}]{s}[/{color}]"

    for r in results:
        j = r.get("scores", {})
        ret_style = "green" if r["retrieval_hit"] else "red"
        pass_mark = "[green]✓[/green]" if r["answer_pass"] else "[red]✗[/red]"

        row = [
            str(r["id"]),
            r["category"],
            r["query"][:22] + ("…" if len(r["query"]) > 22 else ""),
            f"[{ret_style}]{'✓' if r['retrieval_hit'] else '✗'}[/{ret_style}]",
        ]
        if judge_mode == "score":
            f = j.get("faithfulness", {}).get("score", "-")
            c = j.get("correctness",  {}).get("score", "-")
            p = j.get("conciseness",  {}).get("score", "-")
            row += [_styled(f), _styled(c), _styled(p)]
            for key, val in [("faithfulness", f), ("correctness", c), ("conciseness", p)]:
                if isinstance(val, int):
                    dim_totals[key].append(val)

        row += [pass_mark, str(r["ttft_sec"]) if r["ttft_sec"] else "-", str(r["tps"])]
        table.add_row(*row)

        if r["ttft_sec"]:
            ttft_list.append(r["ttft_sec"])
        tps_list.append(r["tps"])

    console.print(table)

    n = len(results)
    ret_acc  = sum(r["retrieval_hit"] for r in results) / n * 100
    pass_acc = sum(r["answer_pass"]   for r in results) / n * 100
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tps  = sum(tps_list)  / len(tps_list)

    label = f"pass = correctness ≥ {PASS_THRESHOLD}" if judge_mode == "score" else "binary judge"
    console.print(f"\n[bold]彙總統計（共 {n} 題，{label}）[/bold]")
    console.print(f"  Retrieval Hit Rate : [cyan]{ret_acc:.1f}%[/cyan]  ({sum(r['retrieval_hit'] for r in results)}/{n})")
    console.print(f"  Answer Pass Rate   : [cyan]{pass_acc:.1f}%[/cyan]  ({sum(r['answer_pass'] for r in results)}/{n})  ← Claude 判斷")
    console.print(f"  Avg TTFT           : [cyan]{avg_ttft:.3f}s[/cyan]")
    console.print(f"  Avg TPS            : [cyan]{avg_tps:.2f}[/cyan]")

    if judge_mode == "score":
        console.print("\n[bold]各維度平均分（滿分 5）[/bold]")
        for dim, vals in dim_totals.items():
            avg = sum(vals) / len(vals) if vals else 0
            color = "green" if avg >= 4 else ("yellow" if avg >= 3 else "red")
            console.print(f"  {dim:15s}: [{color}]{avg:.2f}[/{color}]")

    console.rule("\n[bold]📝 定性分析[/bold]")
    for r in results:
        mark = "[green]✓[/green]" if r["answer_pass"] else "[red]✗[/red]"
        console.print(
            f"  Q{r['id']:2d} {mark}  {r['query'][:36]}\n"
            f"       → {r.get('judge_reason', '')}"
        )


def _build_summary(results: list[dict], judge_mode: str) -> dict:
    n = len(results)
    ttft_list = [r["ttft_sec"] for r in results if r["ttft_sec"]]
    tps_list  = [r["tps"] for r in results]

    summary: dict = {
        "total_questions": n,
        "judge_mode": judge_mode,
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in results) / n * 100, 1),
        "retrieval_hit_count": sum(r["retrieval_hit"] for r in results),
        "answer_pass_rate": round(sum(r["answer_pass"] for r in results) / n * 100, 1),
        "answer_pass_count": sum(r["answer_pass"] for r in results),
        "avg_ttft_sec": round(sum(ttft_list) / len(ttft_list), 3) if ttft_list else None,
        "avg_tps": round(sum(tps_list) / len(tps_list), 2),
    }

    if judge_mode == "score":
        dim_totals: dict[str, list[int]] = {"faithfulness": [], "correctness": [], "conciseness": []}
        for r in results:
            j = r.get("scores", {})
            for dim in dim_totals:
                s = j.get(dim, {}).get("score")
                if isinstance(s, int):
                    dim_totals[dim].append(s)
        summary["avg_scores"] = {
            dim: round(sum(vals) / len(vals), 2) if vals else None
            for dim, vals in dim_totals.items()
        }

    return summary


def _save_results(results: list[dict], model: str, judge_mode: str, hybrid: bool = False) -> None:
    summary = _build_summary(results, judge_mode)
    output = {"summary": summary, "results": results}
    retriever = "hybrid" if hybrid else "semantic"
    out = Path(f"data/judge_results_{model}_{retriever}_{judge_mode}.json")
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[dim]結果已儲存至 {out}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["3b", "7b"],            default="3b")
    parser.add_argument("--top-k",      type=int,                         default=3)
    parser.add_argument("--judge-mode", choices=["binary", "score"],      default="binary",
                        help="binary（預設）：correct/incorrect；score：三維度 1-5 分")
    parser.add_argument("--hybrid",     action="store_true",
                        help="使用 Hybrid Retriever（BM25 + FAISS + RRF）")
    args = parser.parse_args()
    run_benchmark_with_judge(
        top_k=args.top_k, model=args.model, judge_mode=args.judge_mode, hybrid=args.hybrid
    )
