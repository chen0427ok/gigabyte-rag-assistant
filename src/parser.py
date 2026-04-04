"""
Step 2 - Phase 2: BeautifulSoup parser for GIGABYTE spec matrix table.

Table structure (multi-model comparison):
  <tr>
    <td>屬性 (Key)</td>
    <td>Model A value</td>
    <td>Model B value</td>
    ...
  </tr>

Output format:
  {
    "AM6H-ModelA": {"作業系統": "...", "中央處理器": "...", ...},
    "AM6H-ModelB": {...},
    ...
  }
"""

import json
from pathlib import Path
from bs4 import BeautifulSoup

INPUT_HTML = Path("data/raw/aorus_master_16.html")
OUTPUT_JSON = Path("data/raw/specs.json")


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse(html: str) -> dict[str, dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Locate the spec section ──────────────────────────────────────────
    spec_section = (
        soup.find(id="Specification")
        or soup.find(class_="spec-table")
        or soup.find(class_="specification")
    )

    if spec_section is None:
        # Fallback: find the table that has the most rows
        all_tables = soup.find_all("table")
        if not all_tables:
            raise RuntimeError("No spec table found in HTML. Check the saved HTML file.")
        spec_section = max(all_tables, key=lambda t: len(t.find_all("tr")))
        print(f"[parser] Fallback: using largest table ({len(spec_section.find_all('tr'))} rows)")

    table = spec_section.find("table") if spec_section.name != "table" else spec_section
    if table is None:
        raise RuntimeError("Spec section found but contains no <table>.")

    rows = table.find_all("tr")
    print(f"[parser] Found {len(rows)} rows in spec table")

    # ── 2. Extract model names from the header row ───────────────────────────
    header_row = rows[0]
    header_cells = header_row.find_all(["th", "td"])

    # First cell is usually empty or "Spec" label; the rest are model names
    model_names: list[str] = []
    for cell in header_cells[1:]:
        name = _clean(cell.get_text())
        if name:
            model_names.append(name)

    if not model_names:
        # No header row — assign generic names
        # Detect column count from the first data row
        first_data = rows[1] if len(rows) > 1 else rows[0]
        col_count = len(first_data.find_all(["th", "td"])) - 1
        model_names = [f"Model_{i+1}" for i in range(col_count)]
        data_rows = rows
    else:
        data_rows = rows[1:]

    print(f"[parser] Models detected: {model_names}")

    # ── 3. Build specs dict ──────────────────────────────────────────────────
    specs: dict[str, dict[str, str]] = {model: {} for model in model_names}

    for row in data_rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        key = _clean(cells[0].get_text())
        if not key:
            continue

        for i, model in enumerate(model_names):
            if i + 1 < len(cells):
                value = _clean(cells[i + 1].get_text())
                if value:
                    specs[model][key] = value

    return specs


def save_json(specs: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    total_keys = sum(len(v) for v in specs.values())
    print(f"[parser] JSON saved → {OUTPUT_JSON}  ({len(specs)} models, {total_keys} total spec entries)")


if __name__ == "__main__":
    if not INPUT_HTML.exists():
        raise FileNotFoundError(f"HTML not found: {INPUT_HTML}. Run scraper.py first.")

    html = INPUT_HTML.read_text(encoding="utf-8")
    specs = parse(html)

    # Preview
    for model, data in specs.items():
        print(f"\n── {model} ({len(data)} specs) ──")
        for k, v in list(data.items())[:5]:
            print(f"  {k}: {v}")
        if len(data) > 5:
            print(f"  ... (+{len(data)-5} more)")

    save_json(specs)
