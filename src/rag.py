"""
Step 7 - Complete RAG Pipeline

整合 Retriever + Generator，提供互動式問答介面。

使用方式：
    uv run python src/rag.py                  # 互動模式
    uv run python src/rag.py --query "BZH 的 GPU？"  # 單次查詢
"""

import argparse
from rich.console import Console
from rich.panel import Panel

from retriever import Retriever
from hybrid_retriever import HybridRetriever
from generator import Generator

console = Console()


class RAGPipeline:
    def __init__(self, top_k: int = 3, hybrid: bool = False):
        console.print("[bold cyan]載入 Retriever...[/bold cyan]")
        if hybrid:
            self.retriever = HybridRetriever(top_k=top_k)
            console.print("[dim]  模式：Hybrid（BM25 + FAISS + RRF）[/dim]")
        else:
            self.retriever = Retriever(top_k=top_k)
            console.print("[dim]  模式：Dense（FAISS only）[/dim]")
        console.print("[bold cyan]載入 Generator...[/bold cyan]")
        self.generator = Generator()
        console.print("[bold green]RAG Pipeline 就緒！[/bold green]\n")

    def ask(self, query: str) -> dict:
        # Step 1: Retrieve
        chunks = self.retriever.retrieve(query)

        console.print("\n[bold yellow]📎 Retrieved Chunks:[/bold yellow]")
        for i, c in enumerate(chunks, 1):
            console.print(
                f"  [{i}] score=[cyan]{c['score']}[/cyan] | "
                f"[magenta]{c['model']}[/magenta] | {c['key']}"
            )

        # Step 2: Generate
        console.print("\n[bold yellow]🤖 回答：[/bold yellow]", end="")
        result = self.generator.generate(query, chunks)

        # Step 3: Show metrics
        m = result["metrics"]
        console.print(
            f"\n[dim]📊 TTFT=[cyan]{m['ttft_sec']}s[/cyan]  "
            f"TPS=[cyan]{m['tps']}[/cyan]  "
            f"tokens=[cyan]{m['total_tokens']}[/cyan]  "
            f"total=[cyan]{m['total_sec']}s[/cyan][/dim]"
        )

        return result


def interactive_mode(pipeline: RAGPipeline) -> None:
    console.print(Panel(
        "[bold]GIGABYTE AORUS MASTER 16 RAG 小幫手[/bold]\n"
        "輸入問題後按 Enter，輸入 [red]quit[/red] 或 [red]q[/red] 離開",
        border_style="cyan",
    ))

    while True:
        try:
            query = console.input("\n[bold green]問題> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "q", "exit"):
            console.print("[dim]bye![/dim]")
            break

        pipeline.ask(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="GIGABYTE RAG Assistant")
    parser.add_argument("--query", type=str, default=None, help="單次查詢模式")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieve 幾個 chunks（預設 3）")
    parser.add_argument("--hybrid", action="store_true", help="使用 Hybrid Retriever（BM25 + FAISS + RRF）")
    args = parser.parse_args()

    pipeline = RAGPipeline(top_k=args.top_k, hybrid=args.hybrid)

    if args.query:
        pipeline.ask(args.query)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
