"""
Step 2 - Phase 1: Playwright scraper for GIGABYTE AORUS MASTER 16 AM6H
Launches headless Chromium, waits for spec table to load, extracts raw HTML.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

TARGET_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H"
OUTPUT_DIR = Path("data/raw")
OUTPUT_HTML = OUTPUT_DIR / "aorus_master_16.html"


async def scrape() -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="zh-TW",
        )
        page = await context.new_page()

        print(f"[scraper] Navigating to {TARGET_URL}")
        await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)

        # Click on "產品規格" tab to load spec content
        spec_tab_candidates = [
            "a[href='#Specification']",
            "a[href='#kf']",
            "li a:has-text('規格')",
            "a:has-text('產品規格')",
            "a:has-text('Specification')",
            ".tab-item:has-text('規格')",
        ]
        clicked = False
        for tab_sel in spec_tab_candidates:
            try:
                await page.wait_for_selector(tab_sel, timeout=8_000)
                await page.click(tab_sel)
                print(f"[scraper] Clicked spec tab via: '{tab_sel}'")
                clicked = True
                break
            except Exception:
                print(f"[scraper] Tab selector not found: '{tab_sel}', trying next...")

        if not clicked:
            print("[scraper] WARNING: Could not click spec tab, proceeding anyway.")

        # Wait for spec table to appear after tab click
        spec_candidates = [
            "#Specification table",
            "#kf table",
            ".spec-table table",
            "table",
        ]
        matched = None
        for selector in spec_candidates:
            try:
                await page.wait_for_selector(selector, timeout=15_000)
                matched = selector
                print(f"[scraper] Spec table found via: '{selector}'")
                break
            except Exception:
                print(f"[scraper] Selector not found: '{selector}', trying next...")

        if matched is None:
            print("[scraper] WARNING: No spec table matched, saving page as-is.")

        html = await page.content()
        await browser.close()

    return html


def save_html(html: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size // 1024
    print(f"[scraper] HTML saved → {OUTPUT_HTML}  ({size_kb} KB)")


if __name__ == "__main__":
    html = asyncio.run(scrape())
    save_html(html)
