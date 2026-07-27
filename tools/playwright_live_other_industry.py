#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = "http://127.0.0.1:8080/synthesis.html"
API_URL = "http://127.0.0.1:8000/healthz"


def log(step: int, action: str, detail: str = "") -> int:
    if detail:
        print(f"[{step:02d}] {action}: {detail}", flush=True)
    else:
        print(f"[{step:02d}] {action}", flush=True)
    return step + 1


def best_answer_for_question(question_text: str, repeat_count: int = 0) -> str:
    q = question_text.lower()
    if any(tok in q for tok in ("sales", "revenue", "orders", "volume", "demand")):
        if repeat_count == 0:
            return "Weekly totals per coffee shop are usually between 120 and 220 orders, averaging about 165."
        return "More specifically: Shop A averages 210 orders/week, Shop B 170, Shop C 140, Shop D 120."
    if any(tok in q for tok in ("advance", "lead time", "between order and delivery", "delivery")):
        if repeat_count == 0:
            return "Most clients place orders about 2 days before delivery, with urgent requests same day."
        return "Typical lead time is 48 hours; about 20% of orders are urgent and arrive within 6-12 hours."
    if any(tok in q for tok in ("season", "promotion", "holiday", "festival", "weather")):
        if repeat_count == 0:
            return "Yes. Demand rises about 25 to 35 percent during holidays and local events."
        return "Yes. Peaks happen in November and December, with roughly 30 percent higher weekly volume."
    if any(tok in q for tok in ("inventory", "stock", "visibility", "real-time", "track")):
        if repeat_count == 0:
            return "We have partial visibility. About 70% of clients share daily stock reports, 30% share weekly."
        return "We use a shared sheet plus POS exports. Real-time visibility exists for top clients only."
    if any(tok in q for tok in ("constraint", "limit", "capacity", "budget", "team", "resource")):
        if repeat_count == 0:
            return "Warehouse capacity is about 4,500 orders per day and overtime budget is capped at $25,000 per month."
        return "Main constraints are dock capacity, driver availability, and a fixed monthly logistics budget."
    if any(tok in q for tok in ("kpi", "success", "goal", "target", "objective")):
        return "Success means on-time delivery above 97%, stockout rate below 3%, and margin improvement of at least 5%."
    if repeat_count == 0:
        return "Current baseline is stable with moderate weekly swings; I can provide more specific numbers if needed."
    return "More detail: baseline demand is stable, with short spikes around events and occasional urgent reorder batches."


def parse_question_index(text: str) -> tuple[int, int]:
    m = re.search(r"question\s+(\d+)\s+of\s+(\d+)", text, flags=re.IGNORECASE)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def latest_assistant_message(page) -> str:
    nodes = page.locator("#messages .message.assistant")
    count = nodes.count()
    if count <= 0:
        return ""
    return nodes.nth(count - 1).inner_text().strip()


def latest_question_message(page) -> str:
    nodes = page.locator("#messages .message.assistant")
    for idx in range(nodes.count() - 1, -1, -1):
        text = nodes.nth(idx).inner_text().strip()
        if re.search(r"question\s+\d+\s+of\s+\d+", text, flags=re.IGNORECASE):
            return text
    return ""


def wait_for_next_step(page, last_assistant_count: int, timeout_sec: float = 120.0) -> tuple[str, int, bool]:
    """Wait for either a new guided question or guided-completion progression."""
    deadline = time.time() + timeout_sec
    completion_markers = (
        "working on your analysis",
        "analysis will start",
        "view data dashboard",
        "dashboard",
    )
    while time.time() < deadline:
        if "dashboard.html" in page.url.lower():
            return ("", last_assistant_count, True)
        nodes = page.locator("#messages .message.assistant")
        count = nodes.count()
        if count > last_assistant_count:
            new_texts: list[str] = []
            for idx in range(last_assistant_count, count):
                txt = nodes.nth(idx).inner_text().strip()
                if not txt:
                    continue
                new_texts.append(txt)
                if re.search(r"question\s+\d+\s+of\s+\d+", txt, flags=re.IGNORECASE):
                    return (txt, count, False)
            if new_texts:
                joined = " ".join(t.lower() for t in new_texts)
                if any(marker in joined for marker in completion_markers):
                    return ("", count, True)
            last_assistant_count = count
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for next Toji step.")


def maybe_log_working_chip(page, step: int) -> int:
    try:
        chip = page.locator("#reportChip")
        if chip.count() <= 0:
            return step
        txt = chip.first.inner_text(timeout=2000).strip()
        if txt:
            return log(step, "CHIP", txt)
    except Exception:
        return step
    return step


def main() -> int:
    step = 1
    try:
        health = requests.get(API_URL, timeout=10)
        payload = health.json() if health.ok else {}
        step = log(step, "HEALTH", f"status={health.status_code} body={payload}")
    except Exception as exc:
        step = log(step, "HEALTH", f"failed ({exc})")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=False, slow_mo=250)
        context = browser.new_context(viewport={"width": 1440, "height": 980})
        page = context.new_page()

        step = log(step, "NAVIGATE", APP_URL)
        page.goto(APP_URL, wait_until="domcontentloaded")

        other_card = page.locator('.industry-option[data-industry="__custom__"]')
        step = log(step, "CLICK", "Other Industry")
        other_card.click()

        custom_input = page.locator("#customIndustryInput")
        custom_value = "coffee_distribution_services"
        step = log(step, "TYPE", f"custom industry -> {custom_value}")
        custom_input.fill(custom_value)

        use_btn = page.get_by_role("button", name="Use")
        step = log(step, "CLICK", "Use custom industry")
        use_btn.click()

        problem = (
            "We distribute coffee to independent shops and are missing delivery lead-time visibility, "
            "causing rush orders, stockouts, and margin pressure. I need a practical 30-day plan."
        )
        problem_box = page.locator("#problemStatement")
        step = log(step, "TYPE", "business problem statement")
        problem_box.fill(problem)

        submit = page.locator("#submitBtn")
        step = log(step, "CLICK", "Continue with Toji")
        submit.click()

        try:
            page.wait_for_url(re.compile(r".*/chat\.html.*"), timeout=90000)
        except PlaywrightTimeoutError as exc:
            raise TimeoutError("Did not reach chat page after submit.") from exc
        step = log(step, "PAGE", page.url)

        repeat_by_question: dict[str, int] = defaultdict(int)
        asked_total = 0
        current_idx = 0
        max_turns = 12
        assistant_count = page.locator("#messages .message.assistant").count()
        question_text = latest_question_message(page)
        guided_completed = False

        for _ in range(max_turns):
            if not question_text:
                question_text, assistant_count, guided_completed = wait_for_next_step(page, assistant_count, timeout_sec=120.0)
                if guided_completed:
                    break
            idx, total = parse_question_index(question_text)
            if idx:
                asked_total = max(asked_total, total)
                current_idx = idx
            question_key = re.sub(r"\s+", " ", question_text.lower())
            repeat_count = repeat_by_question[question_key]
            step = log(step, "READ", f"question {current_idx}/{asked_total}: {question_text.splitlines()[0]}")

            answer = best_answer_for_question(question_text, repeat_count=repeat_count)
            repeat_by_question[question_key] += 1
            step = log(step, "TYPE", f"answer -> {answer}")
            page.locator("#input").fill(answer)
            step = log(step, "CLICK", "Send")
            page.locator("#sendBtn").click()
            step = maybe_log_working_chip(page, step)

            # Wait for next step (next question or analysis completion).
            question_text, assistant_count, guided_completed = wait_for_next_step(page, assistant_count, timeout_sec=120.0)
            if guided_completed:
                break
            # Stop only if Toji indicates completion; otherwise keep consuming questions.

        # After guided intake, allow pipeline to run and attempt dashboard navigation.
        step = log(step, "WAIT", "analysis completion / possible dashboard redirect")
        end_wait = time.time() + 180
        while time.time() < end_wait:
            url = page.url
            if "dashboard.html" in url:
                step = log(step, "PAGE", f"dashboard reached -> {url}")
                break
            last_msg = latest_assistant_message(page).lower()
            if "view data dashboard" in page.content().lower():
                try:
                    page.get_by_role("button", name=re.compile("View Data Dashboard", re.IGNORECASE)).click()
                    page.wait_for_url(re.compile(r".*/dashboard\.html.*"), timeout=20000)
                    step = log(step, "CLICK", "View Data Dashboard")
                    step = log(step, "PAGE", f"dashboard reached -> {page.url}")
                    break
                except Exception:
                    pass
            if "could not complete" in last_msg or "unavailable" in last_msg:
                step = log(step, "ASSISTANT", latest_assistant_message(page))
                break
            time.sleep(1.0)

        # If still on chat and guided flow ended, ask a free-form CEO question.
        if "chat.html" in page.url:
            ceo_prompt = "Summarize my top 3 business risks and the first actions I should take this week."
            step = log(step, "TYPE", f"follow-up -> {ceo_prompt}")
            page.locator("#input").fill(ceo_prompt)
            step = log(step, "CLICK", "Send")
            page.locator("#sendBtn").click()
            step = maybe_log_working_chip(page, step)
            time.sleep(3.0)
            step = log(step, "ASSISTANT", latest_assistant_message(page))

        step = log(step, "DONE", "live run finished; browser stays open for manual review")
        # Keep browser open so user can watch.
        while True:
            time.sleep(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
