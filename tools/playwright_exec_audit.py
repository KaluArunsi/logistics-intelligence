#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_URL = "http://127.0.0.1:8080"
SUPPORTED_UI_INDUSTRIES = {"ecommerce", "shipping_freight", "trucking_delivery"}


def load_categories() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    ui_rows: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for path in sorted((ROOT / "config" / "router" / "manifests").glob("*_router_manifest.json")):
        payload = json.loads(path.read_text())
        industry = str(payload.get("industry") or path.name.replace("_router_manifest.json", "")).strip()
        for row in payload.get("categories") or []:
            cat = str((row or {}).get("category") or "").strip()
            if not cat:
                continue
            if industry in SUPPORTED_UI_INDUSTRIES:
                ui_rows.append((industry, cat))
            else:
                skipped.append((industry, cat))
    return ui_rows, skipped


def select_first_nonempty(page, selector: str) -> str:
    values = page.eval_on_selector_all(f"{selector} option", "opts => opts.map(o => o.value)")
    chosen = ""
    for v in values:
        if str(v).strip():
            chosen = str(v)
            break
    if chosen:
        page.select_option(selector, chosen)
    return chosen


def wait_for_url_contains(page, needles: list[str], timeout_ms: int = 300000) -> bool:
    start = time.time()
    timeout_sec = timeout_ms / 1000.0
    while (time.time() - start) < timeout_sec:
        url = page.url
        if any(n in url for n in needles):
            return True
        # Bail out early if synthesis surfaced a visible error.
        if page.locator("#errorMessage.show").count() > 0:
            return False
        page.wait_for_timeout(300)
    return False


def send_chat(page, text: str, timeout_ms: int = 45000) -> str:
    start = time.time()
    timeout_s = timeout_ms / 1000.0
    # Wait until chat composer is present.
    while (time.time() - start) < timeout_s:
        if "chat.html" not in page.url:
            return ""
        if page.locator("#input").count() == 0 or page.locator("#sendBtn").count() == 0:
            page.wait_for_timeout(250)
            continue
        break
    if "chat.html" not in page.url:
        return ""
    if page.locator("#sendBtn").count() == 0:
        return ""

    assistant_count = page.locator(".message.assistant").count()
    try:
        page.locator("#input").fill(text)
        # UI enables send button after input event handling.
        waited = 0.0
        while waited < 4.0 and not page.locator("#sendBtn").is_enabled():
            page.wait_for_timeout(100)
            waited += 0.1
        if not page.locator("#sendBtn").is_enabled():
            return ""
        page.locator("#sendBtn").click()
    except Exception:
        return ""
    start = time.time()
    while (time.time() - start) < (timeout_ms / 1000.0):
        if "chat.html" not in page.url:
            return ""
        new_assistant_count = page.locator(".message.assistant").count()
        if new_assistant_count > assistant_count:
            break
        page.wait_for_timeout(250)
    assistants = page.locator(".message.assistant")
    if assistants.count() == 0:
        return ""
    return assistants.nth(assistants.count() - 1).inner_text().strip()


def q_category_text(category: str) -> str:
    return category.replace("_", " ")


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def select_category_chip(page, category: str, timeout_ms: int = 12000) -> bool:
    target = _norm_text(category)
    chips = page.locator("#suggestedCategories .suggest-chip")
    start = time.time()
    timeout_s = timeout_ms / 1000.0
    while (time.time() - start) < timeout_s:
        count = chips.count()
        if count > 0:
            for idx in range(count):
                text = chips.nth(idx).inner_text().strip()
                if _norm_text(text) == target or _norm_text(text.replace(" ", "_")) == target:
                    chips.nth(idx).click()
                    return True
            # fallback: contains match
            for idx in range(count):
                text = chips.nth(idx).inner_text().strip()
                if target in _norm_text(text) or _norm_text(text) in target:
                    chips.nth(idx).click()
                    return True
            return False
        page.wait_for_timeout(250)
    return False


def resolve_report_id(page) -> str | None:
    parsed = urlparse(page.url)
    rid = (parse_qs(parsed.query).get("report_id") or [None])[0]
    if rid:
        return str(rid)
    try:
        rid = page.evaluate(
            """async () => {
                try {
                    const api = window.location.port === '8080' ? 'http://localhost:8000' : '';
                    const res = await fetch(`${api}/session/status`);
                    if (!res.ok) return null;
                    const j = await res.json();
                    return j && j.report_id ? String(j.report_id) : null;
                } catch (_) {
                    return null;
                }
            }"""
        )
        return str(rid) if rid else None
    except Exception:
        return None


def run_one(browser, industry: str, category: str, idx: int) -> dict[str, Any]:
    forwarded_ip = f"127.0.{(idx // 255) % 255}.{idx % 255}"
    ctx = browser.new_context(
        viewport={"width": 1440, "height": 900},
        extra_http_headers={"x-forwarded-for": forwarded_ip},
    )
    page = ctx.new_page()

    result: dict[str, Any] = {
        "industry": industry,
        "target_category": category,
        "actual_category": None,
        "report_id": None,
        "tell_us_submitted": False,
        "guided_mode": False,
        "unsat_reframed": False,
        "guided_completed": False,
        "dashboard_loaded": False,
        "forecast_panel": False,
        "trend_panel": False,
        "visual_count": 0,
        "visuals_error": None,
        "toji_answer_quality": False,
        "toji_offtopic_guard": False,
        "issues": [],
    }

    try:
        page.goto(f"{FRONTEND_URL}/synthesis.html", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_selector("#synthesisForm", timeout=20000)

        # Industry button click (actual user interaction)
        page.locator(f".industry-option[data-industry='{industry}']").click()
        page.wait_for_timeout(400)
        select_category_chip(page, category)

        cat_text = q_category_text(category)
        page.locator("#q1").fill(f"We need to optimize {cat_text} immediately.")
        page.locator("#q2").fill("United States, Europe, and Asia corridors.")
        select_first_nonempty(page, "#q3")
        page.locator("#q4").fill(
            f"Primary bottleneck is in {cat_text}. This has been happening for the past 4 months and is now affecting customer outcomes."
        )
        select_first_nonempty(page, "#q5")
        select_first_nonempty(page, "#q6")
        page.locator("#q7").fill("Past 4 months under budget and staffing constraints; we need low-disruption interventions first.")
        page.locator("#q8").fill("7")
        page.locator("#q9").fill("Improve primary KPI by 10-15% with no SLA regression.")

        page.locator("#submitBtn").click()
        result["tell_us_submitted"] = True

        if not wait_for_url_contains(page, ["dashboard.html", "chat.html"], timeout_ms=420000):
            result["issues"].append("timeout_waiting_for_dashboard_or_chat")
            return result

        # Some runs land on dashboard and auto-redirect to guided chat.
        page.wait_for_timeout(1500)

        if "dashboard.html" in page.url:
            result["dashboard_loaded"] = True
            try:
                page.wait_for_selector("#industryTitle", timeout=8000)
                title = page.locator("#industryTitle").inner_text().strip()
                result["actual_category"] = title.split("/")[-1].strip() if "/" in title else title
            except Exception:
                # Keep run going: title hydration can lag under heavy browser churn.
                pass

            result["forecast_panel"] = page.locator("text=Future Forecast").count() > 0
            result["trend_panel"] = page.locator("text=Trend Visualization").count() > 0

            # Click to Toji from dashboard.
            if page.locator("#openTojiLink").count() > 0:
                page.locator("#openTojiLink").click()
                wait_for_url_contains(page, ["chat.html"], timeout_ms=30000)

        if "chat.html" not in page.url:
            result["issues"].append("did_not_reach_chat")
            return result

        # If report id present, keep it.
        result["report_id"] = resolve_report_id(page)

        # Guided mode detection + unsatisfactory answer reframe check.
        page.wait_for_timeout(1500)
        assistant_texts = page.eval_on_selector_all(
            ".message.assistant",
            "els => els.map(e => (e.innerText || e.textContent || '').trim())",
        )
        guided_hint = any(re.search(r"\[\s*\d+\s+of\s+\d+\s*\]", str(t).lower()) for t in assistant_texts)
        result["guided_mode"] = guided_hint or ("guided_missing=1" in page.url) or ("intake_mode=1" in page.url)

        if result["guided_mode"]:
            r0 = send_chat(page, "i don't know")
            low = r0.lower()
            if any(tok in low for tok in ["no problem", "rough estimate", "that is okay", "even a rough estimate"]):
                result["unsat_reframed"] = True

            # Complete guided flow with usable answers. Wait for a real report id.
            deadline = time.time() + 120
            while time.time() < deadline:
                if "dashboard.html" in page.url:
                    result["guided_completed"] = True
                    break
                if "chat.html" not in page.url:
                    page.wait_for_timeout(500)
                    continue
                can_send = (
                    page.locator("#sendBtn").count() > 0
                    and page.locator("#sendBtn").is_enabled()
                )
                if not can_send:
                    # Likely running guided rerun in the background.
                    page.wait_for_timeout(750)
                    rid = resolve_report_id(page)
                    if rid:
                        result["report_id"] = rid
                    continue
                resp = send_chat(
                    page,
                    f"This has been happening for 4 months. It mainly affects {cat_text} and worsens during peak weeks.",
                    timeout_ms=15000,
                )
                if "dashboard is ready" in resp.lower():
                    result["guided_completed"] = True
                    break
                if any(tok in resp.lower() for tok in ["analysis is ready", "taking you to the dashboard", "refreshed dashboard is ready"]):
                    result["guided_completed"] = True
                rid = resolve_report_id(page)
                if rid:
                    result["report_id"] = rid

            # Guided flow can auto-redirect to dashboard when complete.
            if "dashboard.html" in page.url and not result["guided_completed"]:
                result["guided_completed"] = True

        # Refresh resolved report id after guided phase.
        if not result["report_id"]:
            result["report_id"] = resolve_report_id(page)

        # Ask Toji about the generated report.
        if "chat.html" not in page.url and "dashboard.html" in page.url:
            if page.locator("#openTojiLink").count() > 0:
                page.locator("#openTojiLink").click()
                wait_for_url_contains(page, ["chat.html"], timeout_ms=30000)

        a1 = send_chat(page, "Summarize the top three operational risks in this dashboard.")
        a2 = send_chat(page, "What should we do in the next 30 days to improve outcomes?")
        a3 = send_chat(page, "Are you gpt or groq?")

        joined = "\n".join([a1, a2]).lower()
        result["toji_answer_quality"] = bool(a1.strip() and a2.strip()) and ("unavailable" not in joined)

        a3l = a3.lower()
        leaks = any(tok in a3l for tok in ["gpt-oss", "groq", "llama", "openai model", "provider"])
        result["toji_offtopic_guard"] = (not leaks) and bool(a3.strip())

        # Back to dashboard via button click.
        if page.locator("button:has-text('Back to dashboard')").count() > 0:
            page.locator("button:has-text('Back to dashboard')").click()
            wait_for_url_contains(page, ["dashboard.html"], timeout_ms=30000)

        if "dashboard.html" in page.url:
            result["dashboard_loaded"] = True
            page.wait_for_timeout(1200)
            result["forecast_panel"] = page.locator("text=Future Forecast").count() > 0
            result["trend_panel"] = page.locator("text=Trend Visualization").count() > 0
            result["visual_count"] = page.locator("#visualsWrapper img").count()
            if page.locator("#visualDiagnostics").count() > 0:
                diag = page.locator("#visualDiagnostics").inner_text().strip()
                if "issue detected" in diag.lower() or "failed" in diag.lower():
                    result["visuals_error"] = diag
            if not result["actual_category"]:
                try:
                    page.wait_for_selector("#industryTitle", timeout=8000)
                    title = page.locator("#industryTitle").inner_text().strip()
                    result["actual_category"] = title.split("/")[-1].strip() if "/" in title else title
                except Exception:
                    pass

    except PlaywrightTimeoutError as exc:
        result["issues"].append(f"timeout:{exc}")
    except Exception as exc:
        result["issues"].append(f"exception:{type(exc).__name__}:{exc}")
    finally:
        ctx.close()

    # Coherence checks
    if result["tell_us_submitted"] and not result["dashboard_loaded"]:
        result["issues"].append("dashboard_not_loaded")
    if not result.get("report_id"):
        result["issues"].append("report_id_not_materialized")
    if result["guided_mode"] and not result["unsat_reframed"]:
        result["issues"].append("unsat_input_not_reframed")
    if not result["toji_answer_quality"]:
        result["issues"].append("toji_answer_quality_failed")
    if not result["toji_offtopic_guard"]:
        result["issues"].append("toji_offtopic_guard_failed")

    return result


def write_report(rows: list[dict[str, Any]], skipped: list[tuple[str, str]]) -> tuple[Path, Path]:
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"playwright_exec_audit_{stamp}.json"
    out_md = out_dir / f"playwright_exec_audit_{stamp}.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frontend_url": FRONTEND_URL,
        "ui_supported_total": len(rows),
        "skipped_total": len(skipped),
        "skipped_categories": [{"industry": i, "category": c} for i, c in skipped],
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = []
    lines.append("# Playwright Executive UX Audit\n")
    lines.append(f"Generated: `{payload['generated_at']}`")
    lines.append(f"UI-supported categories tested: **{len(rows)}**")
    lines.append(f"Skipped (industry not available in Tell Us UI): **{len(skipped)}**\n")

    lines.append("| Industry | Target Category | Actual Category | Reframed Unsat | Guided Done | Trend | Forecast | Visuals | Toji Quality | Toji Guard | Issues |")
    lines.append("|---|---|---|---|---|---|---|---:|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['industry']} | {r['target_category']} | {r.get('actual_category') or ''} | "
            f"{r['unsat_reframed']} | {r['guided_completed']} | {r['trend_panel']} | {r['forecast_panel']} | {r['visual_count']} | "
            f"{r['toji_answer_quality']} | {r['toji_offtopic_guard']} | {'; '.join(r['issues'])} |"
        )

    issues = [r for r in rows if r["issues"]]
    lines.append("\n## Summary\n")
    lines.append(f"- Clean runs: **{len(rows) - len(issues)} / {len(rows)}**")
    lines.append(f"- Runs with issues: **{len(issues)}**")
    lines.append(f"- Unsatisfactory input successfully reframed: **{sum(1 for r in rows if r['unsat_reframed'])}**")
    lines.append(f"- Toji off-topic guard passes: **{sum(1 for r in rows if r['toji_offtopic_guard'])}**")

    if skipped:
        lines.append("\n## Skipped\n")
        for ind, cat in skipped:
            lines.append(f"- `{ind}/{cat}` (industry not present in Tell Us UI options)")

    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def main() -> int:
    ui_categories, skipped = load_categories()
    print(f"Loaded UI categories: {len(ui_categories)}; skipped: {len(skipped)}")
    rows: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            for idx, (industry, category) in enumerate(ui_categories, start=1):
                print(f"[{idx}/{len(ui_categories)}] {industry}/{category} ...", flush=True)
                row = run_one(browser, industry, category, idx)
                rows.append(row)
                print(
                    f"  -> dashboard={row['dashboard_loaded']} reframe={row['unsat_reframed']} "
                    f"guided={row['guided_completed']} visuals={row['visual_count']} issues={len(row['issues'])}",
                    flush=True,
                )
                if row["issues"]:
                    print(f"     issues: {row['issues'][:3]}", flush=True)
        finally:
            browser.close()

    out_json, out_md = write_report(rows, skipped)
    print(f"\nWrote:\n- {out_json}\n- {out_md}")
    print(f"Done. total={len(rows)} issues={sum(1 for r in rows if r['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
