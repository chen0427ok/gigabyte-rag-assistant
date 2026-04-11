"""
Step 2 - Phase 2: BeautifulSoup parser for GIGABYTE spec page (desktop swiper layout).

Page structure (AORUS MASTER 16 AM6H):
  div.desktop-spec-content
    div.multiple-spec-content-wrapper
      div.spec-column            ← spec key labels (作業系統, CPU, ...)
        div.multiple-title (×N)
      div.content-column
        div.multiple-content-swiper
          div.swiper-wrapper
            div.swiper-slide     ← one slide per model (BZH, BYH, BXH)
              div.spec-item-list[data-spec-row="0"]  ← value for row 0
              div.spec-item-list[data-spec-row="1"]
              ...

Model name order:
  span.gbt-main-checkbox-label-text lists models as BXH → BYH → BZH.
  Swiper slides are in the reversed order: BZH → BYH → BXH.

Output format:
  {
    "AORUS MASTER 16 BZH": {"作業系統": "...", "中央處理器": "...", ...},
    "AORUS MASTER 16 BYH": {...},
    "AORUS MASTER 16 BXH": {...}
  }
"""

import json
from pathlib import Path
from bs4 import BeautifulSoup, Tag

INPUT_HTML = Path("data/raw/aorus_master_16.html")
OUTPUT_JSON = Path("data/raw/raw_specs.json")


def _extract_text(tag: Tag) -> str:
    """Convert a tag's content to plain text, using \\n for <br> tags."""
    parts = []
    for node in tag.descendants:
        if isinstance(node, str):
            parts.append(node)
        elif node.name == "br":
            parts.append("\n")
    # Strip each line and drop blanks (handles <br/><br/> double newlines)
    lines = [line.strip() for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def parse(html: str) -> dict[str, dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Locate desktop spec section ──────────────────────────────────────
    desktop = soup.find(class_="desktop-spec-content")
    if desktop is None:
        raise RuntimeError(
            "Could not find .desktop-spec-content in HTML. "
            "Make sure the HTML was saved from the rendered page."
        )

    wrapper = desktop.find(class_="multiple-spec-content-wrapper")
    if wrapper is None:
        raise RuntimeError("Could not find .multiple-spec-content-wrapper.")

    # ── 2. Extract spec keys from the label column ───────────────────────────
    spec_col = wrapper.find(class_="spec-column")
    if spec_col is None:
        raise RuntimeError("Could not find .spec-column.")

    keys = [div.get_text(strip=True) for div in spec_col.find_all(class_="multiple-title")]
    if not keys:
        raise RuntimeError("No spec keys found in .spec-column.")
    print(f"[parser] Spec keys ({len(keys)}): {keys}")

    # ── 3. Extract model names from checkbox labels (reversed = slide order) ─
    label_spans = soup.find_all(class_="gbt-main-checkbox-label-text")
    model_names = [
        span.get_text(strip=True)
        for span in label_spans
        if span.get_text(strip=True) not in ("顯示差異",)
    ]
    # Checkbox order is BXH → BYH → BZH; slides are in reverse (BZH → BYH → BXH)
    model_names = list(reversed(model_names))
    if not model_names:
        raise RuntimeError("Could not find model names from .gbt-main-checkbox-label-text spans.")
    print(f"[parser] Models (slide order): {model_names}")

    # ── 4. Extract values from each swiper slide ─────────────────────────────
    swiper_wrapper = desktop.find(class_="swiper-wrapper")
    if swiper_wrapper is None:
        raise RuntimeError("Could not find .swiper-wrapper inside .desktop-spec-content.")

    slides = swiper_wrapper.find_all(class_="swiper-slide", recursive=False)
    print(f"[parser] Slides found: {len(slides)}")

    if len(slides) != len(model_names):
        print(
            f"[parser] WARNING: {len(slides)} slides vs {len(model_names)} model names — "
            "using min count."
        )

    # ── 5. Build specs dict ──────────────────────────────────────────────────
    specs: dict[str, dict[str, str]] = {model: {} for model in model_names}

    for model, slide in zip(model_names, slides):
        value_divs = slide.find_all(class_="spec-item-list")
        for div in value_divs:
            row_idx = div.get("data-spec-row")
            if row_idx is None:
                continue
            row_idx = int(row_idx)
            if row_idx >= len(keys):
                continue
            key = keys[row_idx]
            span = div.find(class_="swiper-no-swiping")
            if span is None:
                continue
            value = _extract_text(span)
            if value:
                specs[model][key] = value

    return specs


def save_json(specs: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    total_keys = sum(len(v) for v in specs.values())
    print(
        f"[parser] JSON saved → {OUTPUT_JSON}  "
        f"({len(specs)} models, {total_keys} total spec entries)"
    )


if __name__ == "__main__":
    if not INPUT_HTML.exists():
        raise FileNotFoundError(
            f"HTML not found: {INPUT_HTML}\n"
            "Run scraper.py first:\n"
            "  uv run python src/scraper.py --local <path/to/saved.html>\n"
            "  uv run python src/scraper.py   (live scrape)"
        )

    html = INPUT_HTML.read_text(encoding="utf-8")
    specs = parse(html)

    # Preview
    for model, data in specs.items():
        print(f"\n── {model} ({len(data)} specs) ──")
        for k, v in list(data.items())[:3]:
            print(f"  {k}: {v[:60]}{'...' if len(v) > 60 else ''}")
        if len(data) > 3:
            print(f"  ... (+{len(data)-3} more)")

    save_json(specs)
