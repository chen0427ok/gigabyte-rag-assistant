"""
feature/llm-judge - Claude API 自動評分

對 RAG 系統生成的答案進行定性評分，補充 evaluate.py 的關鍵字比對。

評分維度（每項 1-5 分）：
  - faithfulness  : 答案是否忠實於 retrieved chunks，無幻覺
  - correctness   : 答案是否正確回應問題
  - conciseness   : 答案是否精簡不冗餘

使用方式：
    uv run python src/llm_judge.py --input data/eval_results_3b.json
    uv run python src/llm_judge.py --input data/eval_results_7b.json
    uv run python src/llm_judge.py --input data/eval_results_3b.json --output data/judge_results_3b.json
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

load_dotenv()

console = Console()

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM = """你是一個嚴格的 RAG 系統評審。
你會收到：使用者問題、參考規格資料（retrieved chunks）、RAG 系統的回答。
請根據以下三個維度各給 1-5 分的整數評分，並簡短說明理由（一句話）。

評分維度：
- faithfulness（忠實度）：答案是否完全基於提供的規格資料，沒有捏造或添加資料中沒有的資訊。
- correctness（正確性）：答案是否正確回答了使用者的問題。
- conciseness（精簡度）：答案是否簡潔，沒有不必要的冗餘。

請嚴格以此 JSON 格式回應，不要輸出任何其他文字：
{
  "faithfulness": {"score": <1-5>, "reason": "<一句話>"},
  "correctness":  {"score": <1-5>, "reason": "<一句話>"},
  "conciseness":  {"score": <1-5>, "reason": "<一句話>"}
}"""


def build_judge_prompt(query: str, chunks: list[dict], answer: str) -> str:
    context = "\n".join(f"- {c['text']}" for c in chunks)
    return f"""使用者問題：{query}

參考規格資料：
{context}

RAG 系統回答：
{answer}"""


def judge_one(client: anthropic.Anthropic, query: str, chunks: list[dict], answer: str) -> dict:
    prompt = build_judge_prompt(query, chunks, answer)
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    scores = json.loads(raw)
    return scores


def run_judge(input_path: Path, output_path: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "Your claude api key":
        console.print("[bold red]錯誤：請在 .env 設定有效的 ANTHROPIC_API_KEY[/bold red]")
        return

    client = anthropic.Anthropic(api_key=api_key)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    console.print(f"\n[bold cyan]LLM Judge 開始評分（共 {len(data)} 題）[/bold cyan]\n")

    judged = []
    for item in data:
        qid = item["id"]
        query = item["query"]
        answer = item["answer"]
        chunks = item.get("retrieved_chunks", [])

        console.print(f"  Q{qid} [{item['category']}] {query[:40]}...", end=" ")

        try:
            scores = judge_one(client, query, chunks, answer)
            avg = round(sum(v["score"] for v in scores.values()) / 3, 2)
            console.print(
                f"[green]F={scores['faithfulness']['score']} "
                f"C={scores['correctness']['score']} "
                f"P={scores['conciseness']['score']} "
                f"avg={avg}[/green]"
            )
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[red]解析失敗：{e}[/red]")
            scores = {}
            avg = None

        judged.append({**item, "judge": scores, "judge_avg": avg})
        time.sleep(0.3)  # 避免打爆 rate limit

    _print_judge_report(judged)
    output_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[dim]結果已儲存至 {output_path}[/dim]")


def _print_judge_report(judged: list[dict]) -> None:
    console.rule("\n[bold]🧑‍⚖️ LLM Judge Report[/bold]")

    table = Table(box=box.MARKDOWN, show_header=True, header_style="bold cyan")
    table.add_column("#",          justify="right", width=3)
    table.add_column("Category",   width=10)
    table.add_column("Query",      width=26)
    table.add_column("Faith.",     justify="center", width=7)
    table.add_column("Correct.",   justify="center", width=7)
    table.add_column("Concise.",   justify="center", width=8)
    table.add_column("Avg",        justify="center", width=5)

    score_totals = {"faithfulness": [], "correctness": [], "conciseness": []}

    for r in judged:
        j = r.get("judge", {})
        f = j.get("faithfulness", {}).get("score", "-")
        c = j.get("correctness",  {}).get("score", "-")
        p = j.get("conciseness",  {}).get("score", "-")
        avg = r.get("judge_avg", "-")

        def _style(s):
            if not isinstance(s, int):
                return str(s)
            color = "green" if s >= 4 else ("yellow" if s == 3 else "red")
            return f"[{color}]{s}[/{color}]"

        table.add_row(
            str(r["id"]),
            r["category"],
            r["query"][:24] + ("…" if len(r["query"]) > 24 else ""),
            _style(f), _style(c), _style(p),
            str(avg),
        )

        for key, val in [("faithfulness", f), ("correctness", c), ("conciseness", p)]:
            if isinstance(val, int):
                score_totals[key].append(val)

    console.print(table)

    console.print("\n[bold]平均分數[/bold]")
    for key, vals in score_totals.items():
        if vals:
            console.print(f"  {key:15s}: [cyan]{sum(vals)/len(vals):.2f}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/eval_results_3b.json"),
        help="eval_results JSON 路徑（需包含 retrieved_chunks）",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="輸出 JSON 路徑（預設與 input 同目錄，加 judge_ 前綴）",
    )
    args = parser.parse_args()

    output = args.output or args.input.parent / f"judge_{args.input.name}"
    run_judge(args.input, output)
