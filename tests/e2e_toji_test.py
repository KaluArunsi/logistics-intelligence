#!/usr/bin/env python3
"""
End-to-end Playwright test: simulate an exec for each category across 3 industries.

Flow per category:
  1. synthesis.html → fill Tell Us form → submit
  2. chat.html (intake_mode) → Toji asks column-context questions → user answers
  3. Pipeline runs → redirected to dashboard.html
  4. Navigate to chat.html (report mode) → Toji delivers briefing
  5. User asks 2 follow-up questions about the report → verify Toji stays on topic

Run:
  python tests/e2e_toji_test.py [--industry ecommerce] [--category basket_intelligence] [--headed]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Exec personas: realistic answers per industry for the 9 Tell-Us fields
# ---------------------------------------------------------------------------

EXEC_PERSONAS: dict[str, dict[str, str]] = {
    "ecommerce": {
        "q1": "Cart abandonment and post-purchase retention",
        "q2": "US DTC site and Amazon marketplace, 60% mobile",
        "q3": "Maximize revenue per visitor",
        "q4": "Revenue leaks at checkout — 68% cart abandonment rate, returns eating 14% of gross. We lose most customers between add-to-cart and payment confirmation.",
        "q5": "Moderately volatile",
        "q6": "Revenue and margin erosion",
        "q7": "Ad spend capped at $180K/month, warehouse can only process 4,200 orders/day, customer service team is 12 people",
        "q8": "7",
        "q9": "+15% conversion rate and >97% on-time fulfillment within 90 days",
        "entity_name": "NovaBuy Commerce",
    },
    "shipping_freight": {
        "q1": "Container dwell time at port terminals and drayage handoff delays",
        "q2": "US West Coast to inland distribution, Asia-Pacific transshipment hubs",
        "q3": "Reduce transit delays",
        "q4": "Containers sit 4-6 days at port before drayage pickup. Customs clearance adds another 2 days on 30% of shipments. Demurrage charges hit $1.2M last quarter.",
        "q5": "Highly seasonal",
        "q6": "Service failures and SLA breaches",
        "q7": "Vessel schedule windows are fixed, labor availability drops 40% during peak season, chassis pool is shared with 3 other carriers",
        "q8": "8",
        "q9": "Reduce average port dwell from 5.2 days to 3.0 days, raise on-time delivery to 94%",
        "entity_name": "Pacific Ridge Logistics",
    },
    "trucking_delivery": {
        "q1": "Last-mile route optimization and driver dispatch sequencing",
        "q2": "Urban same-day zones in metro Atlanta and Dallas-Fort Worth, plus suburban next-day clusters",
        "q3": "On-time delivery rate",
        "q4": "Missed delivery windows spike to 18% on Mondays and Fridays. Dispatch assigns routes manually and doesn't account for real-time traffic or driver fatigue hours. Cost per drop is $8.40 versus target of $6.50.",
        "q5": "Highly volatile",
        "q6": "Missed deliveries and SLA failures",
        "q7": "Drivers limited to 11-hour shifts per DOT, fleet of 85 vehicles with 12 needing maintenance, traffic regulations restrict delivery windows in 3 downtown zones",
        "q8": "6",
        "q9": ">97% on-time delivery and cost per drop under $7.00 within 60 days",
        "entity_name": "UrbanFleet Delivery Co",
    },
}

# Category-specific follow-up answers for Toji's column-context questions
# These are realistic business answers an exec would give
COLUMN_ANSWER_BANK: dict[str, list[str]] = {
    # Monetary / cost
    "cost": ["About $1,200 per unit on average", "Roughly $45 per order", "Around $8.40 per delivery drop"],
    "price": ["Average selling price is $67", "List price ranges from $25 to $350", "We charge $12.50 per mile"],
    "revenue": ["$4.2M monthly gross revenue", "About $180K per week", "Last quarter was $14.8M"],
    "spend": ["$180K monthly ad spend", "Fuel costs run about $0.48 per mile", "We spend roughly $320K/month on labor"],
    "fee": ["$2.50 platform fee per transaction", "Demurrage runs about $350 per container per day"],
    "amount": ["Average order value is $89", "Typical shipment value is $4,500"],
    "value": ["Average cart value is $112", "Cargo value per container averages $45K"],
    # Volume / count
    "volume": ["About 4,200 orders per day", "380 containers per week", "Roughly 1,100 deliveries daily"],
    "count": ["Around 850 per week", "We process about 3,500 daily", "Typically 120 per shift"],
    "quantity": ["Average 3.2 items per order", "Usually 15-20 pallets per truck"],
    "orders": ["4,200 orders per day on average", "We handle about 28,000 weekly"],
    "shipment": ["About 380 shipments per week", "We move roughly 1,600 per month"],
    "units": ["About 12,000 units per day across all SKUs"],
    # Rate / percentage
    "rate": ["About 68%", "Currently running at 14%", "Roughly 94% on average"],
    "ratio": ["About 3:1", "Roughly 0.85", "Currently 72%"],
    "percent": ["Around 18%", "About 30% of the time", "Roughly 94%"],
    "bounce": ["Our bounce rate is about 42%"],
    "conversion": ["Current conversion is 2.8%", "We convert about 3.1% of visitors"],
    "churn": ["Monthly churn is about 4.2%", "We lose roughly 5% of subscribers per month"],
    # Duration / time
    "duration": ["Average session is about 4.5 minutes", "Typical transit is 6 days", "Usually takes 2.3 hours"],
    "time": ["About 48 hours on average", "Usually 3-5 business days", "Peak hours are 10am-2pm"],
    "delay": ["Average delay is 2.1 days", "Usually about 45 minutes late", "Delays average 4.8 days at port"],
    "dwell": ["Container dwell averages 5.2 days", "Packages sit about 18 hours before dispatch"],
    # Temporal
    "month": ["Peak months are November and December", "Busiest Q4, slowest Q1"],
    "season": ["Holiday season drives 40% of annual volume", "Summer is our peak for this lane"],
    "day": ["Mondays and Fridays are worst", "Weekdays are 3x weekend volume"],
    "week": ["About 5,500 per week", "Volume drops 25% on holiday weeks"],
    # Categorical / descriptive
    "type": ["Mostly ground shipping, about 15% expedited", "Mix of FTL and LTL, roughly 60/40"],
    "mode": ["Primarily ocean container with drayage handoff", "90% is truck, 10% rail intermodal"],
    "region": ["US West Coast and Southeast", "Metro Atlanta and DFW are primary zones"],
    "channel": ["60% mobile, 25% desktop, 15% marketplace", "Direct sales and 3PL partners"],
    "status": ["About 85% delivered on first attempt", "70% clear customs same day"],
    # Safety / compliance
    "incident": ["About 3 per month on average", "We've had 12 in the last quarter"],
    "violation": ["2 minor violations in the past 6 months", "Zero major violations this year"],
    "compliance": ["We're at 96% compliance rate", "Audit score was 88 out of 100"],
    "score": ["Current score is 78 out of 100", "NPS is 42, down from 51 last quarter"],
    # Generic fallback
    "default": ["Approximately 500 per period", "About average for our industry", "Roughly in the mid-range — I'd say around 65%"],
}

# Follow-up questions an exec would ask Toji after seeing the report
EXEC_FOLLOWUPS: dict[str, list[str]] = {
    "ecommerce": [
        "What's driving the risk score? Is it the missing fields or something in the data itself?",
        "If I could only fix one thing this week to move the needle on conversion, what would you recommend?",
        "Assume we reduce checkout steps from 5 to 3 next sprint. How should that change conversion and abandonment?",
        "Give me a CEO-style action plan: top 3 actions, owner, and metric to track weekly.",
    ],
    "shipping_freight": [
        "Break down the risk drivers for me — which lanes or stages are contributing most to the elevated risk?",
        "What's the single highest-impact action I can take in the next 30 days to reduce port dwell time?",
        "If I prioritize customs pre-clearance on top 5 lanes, what operational impact should I expect in 30 and 90 days?",
        "Give me the top 3 actions with owner and KPI I should review every Monday.",
    ],
    "trucking_delivery": [
        "The match score seems low — what data am I missing that would improve the analysis?",
        "Give me a prioritized action plan for the next 2 weeks to hit the 97% on-time target.",
        "If I re-sequence dispatch windows and add live traffic routing, what should happen to missed delivery rate?",
        "Give me top 3 actions with owner and one KPI per action for weekly exec review.",
    ],
}


def _pick_answer(field_name: str, question_text: str, industry: str) -> str:
    """Pick a realistic answer by parsing the question Toji actually asked."""
    q = question_text.lower()

    # Time context question (always first)
    if "how long" in q and ("happening" in q or "business" in q or "issue" in q):
        return {"ecommerce": "About 6 months, started around August last year",
                "shipping_freight": "Past 4 months, since the peak season started in October",
                "trucking_delivery": "Ongoing for about 3 months, since we expanded the DFW zone"}[industry]

    # Product / item / aisle / department questions (ecommerce basket)
    if any(w in q for w in ("product", "aisle", "department", "item", "sku", "basket")):
        if "department" in q or "aisle" in q and "contain" in q:
            return "Primarily electronics and home goods — about 40% of basket value"
        if "product name" in q or "representative" in q:
            return "Wireless Bluetooth Earbuds, aisle 62 in our catalog system"
        if "most common" in q or "popular" in q:
            return "Phone cases and screen protectors — about 1,800 units per day"
        return "Mix of consumer electronics and home accessories, average 3.4 items per basket"

    # Order / transaction frequency
    if any(w in q for w in ("order", "transaction", "purchase")) and any(w in q for w in ("how many", "frequency", "average", "typical", "per day")):
        return {"ecommerce": "About 4,200 orders per day, average customer orders every 18 days",
                "shipping_freight": "We process about 380 shipments per week across all lanes",
                "trucking_delivery": "Roughly 1,100 deliveries per day across 85 vehicles"}[industry]

    # Customer / lifetime / loyalty questions
    if any(w in q for w in ("customer", "lifetime", "loyalty", "placed", "reorder", "between")):
        return "Typical customer has 8 lifetime orders, about 22 days between purchases"

    # Delivery / shipment / container specifics
    if any(w in q for w in ("container", "dwell", "port", "terminal")):
        return "Average 5.2 days dwell at terminal, worst lanes hit 8 days during peak"
    if any(w in q for w in ("delivery", "deliveries", "drop", "last mile", "last-mile")):
        return "About 1,100 per day, 97% target but currently at 82% on-time"
    if "route" in q or "dispatch" in q:
        return "85 routes daily, manually dispatched, average 14 stops per route"

    # Cost / price / spend / revenue
    if any(w in q for w in ("cost", "price", "spend", "revenue", "margin", "profit", "fee", "charge", "demurrage")):
        return {"ecommerce": "Average order value is $89, cost per acquisition is about $12",
                "shipping_freight": "Average lane cost is $2,800 per container, demurrage at $350/day",
                "trucking_delivery": "Cost per drop is $8.40, fuel at $0.48/mile, target is $6.50/drop"}[industry]

    # Rate / percentage / ratio
    if any(w in q for w in ("rate", "percent", "ratio", "conversion", "abandon", "return", "churn", "bounce")):
        return {"ecommerce": "Cart abandonment is 68%, return rate is 14%, conversion at 2.8%",
                "shipping_freight": "On-time delivery at 88%, damage claim rate is 2.3%",
                "trucking_delivery": "On-time is 82%, missed delivery rate spikes to 18% Mon/Fri"}[industry]

    # Volume / capacity / utilization
    if any(w in q for w in ("volume", "capacity", "utilization", "throughput", "how many")):
        return {"ecommerce": "About 4,200 orders daily, warehouse handles 5,000 max",
                "shipping_freight": "380 containers per week, terminal capacity is 500",
                "trucking_delivery": "Fleet of 85 trucks, running at about 78% utilization"}[industry]

    # Duration / time / transit / lead time
    if any(w in q for w in ("duration", "transit", "lead time", "how long", "days", "hours", "minutes")):
        return {"ecommerce": "Average fulfillment is 2.1 days, customer session about 4.5 minutes",
                "shipping_freight": "Average transit is 14 days port-to-door, customs adds 2 days",
                "trucking_delivery": "Average route takes 6.5 hours, last mile averages 45 minutes"}[industry]

    # Safety / compliance / risk / incident
    if any(w in q for w in ("safety", "compliance", "incident", "violation", "accident", "inspection")):
        return {"ecommerce": "Zero safety incidents, PCI compliance at 100%",
                "shipping_freight": "3 safety incidents last quarter, 96% compliance score",
                "trucking_delivery": "2 minor incidents this quarter, DOT compliance at 94%"}[industry]

    # Temperature / cold chain
    if any(w in q for w in ("temperature", "cold chain", "refriger", "frozen", "perishable")):
        return "We maintain 2-8°C for pharma, -18°C for frozen. About 15% of volume is temp-controlled."

    # Weight / dimensions / size
    if any(w in q for w in ("weight", "dimension", "size", "heavy", "oversize")):
        return {"ecommerce": "Average package is 2.3 lbs, 12x10x6 inches",
                "shipping_freight": "Average container load is 18 metric tons, mostly 40ft containers",
                "trucking_delivery": "Average parcel is 8 lbs, some routes have 30% oversize items"}[industry]

    # Fuel / energy / emissions
    if any(w in q for w in ("fuel", "energy", "emission", "mpg", "gallon", "diesel", "electric")):
        return "Fleet averages 6.2 MPG, fuel costs about $0.48/mile, no EVs yet but evaluating"

    # Driver / workforce / shift / labor
    if any(w in q for w in ("driver", "workforce", "shift", "labor", "staff", "employee", "headcount")):
        return {"ecommerce": "Warehouse team of 45, customer service is 12 people",
                "shipping_freight": "Terminal crew of 120, plus 35 drayage drivers",
                "trucking_delivery": "85 drivers on roster, 8 currently on medical leave, 11-hour DOT max shifts"}[industry]

    # Geography / region / zone / lane
    if any(w in q for w in ("region", "zone", "lane", "market", "geography", "area", "territory")):
        return {"ecommerce": "Primary markets are US DTC (60%), Amazon (25%), mobile app (15%)",
                "shipping_freight": "US West Coast to inland, Asia transshipment, Gulf to EU",
                "trucking_delivery": "Metro Atlanta, DFW, and suburban next-day zones in Southeast"}[industry]

    # Fallback: give a reasonable numeric answer
    return {"ecommerce": "About 500 per day on average, trending up 8% month over month",
            "shipping_freight": "Roughly 150 per week, with seasonal peaks in Q4",
            "trucking_delivery": "Around 200 daily, higher on weekdays, drops 30% on weekends"}[industry]


def run_single_category(
    *,
    industry: str,
    category: str,
    base_url: str,
    headed: bool,
    timeout_ms: int,
    screenshot_dir: Path,
    start_delay_sec: int = 0,
    hold_open_sec: int = 0,
) -> dict:
    """Run the full e2e flow for one industry+category. Returns a result dict."""
    from playwright.sync_api import sync_playwright

    result = {
        "industry": industry,
        "category": category,
        "status": "pending",
        "tell_us": None,
        "category_fields_filled": 0,
        "intake_questions_count": 0,
        "intake_questions_asked": [],
        "intake_answers_given": [],
        "toji_reframes": [],
        "report_id": None,
        "briefing": None,
        "followup_exchanges": [],
        "errors": [],
        "duration_sec": 0.0,
    }
    t0 = time.time()
    persona = EXEC_PERSONAS[industry]
    cat_label = category.replace("_", " ")

    with sync_playwright() as pw:
        launch_kwargs: dict[str, object] = {"headless": not headed}
        if headed:
            # Prefer installed Chrome for visible desktop UX playback.
            launch_kwargs["channel"] = os.getenv("PW_BROWSER_CHANNEL", "chrome")
            launch_kwargs["slow_mo"] = int(os.getenv("PW_SLOW_MO_MS", "220"))
        try:
            browser = pw.chromium.launch(**launch_kwargs)
        except Exception:
            # Fallback to bundled Chromium if channel browser is unavailable.
            launch_kwargs.pop("channel", None)
            browser = pw.chromium.launch(**launch_kwargs)
        # Use unique X-Forwarded-For IP per category so backend creates distinct sessions.
        # The IP hash = f(secret, ip, coarse_ua, epoch_bucket).
        cat_idx = hash(f"{industry}:{category}") % 250
        fake_ip = f"10.99.{cat_idx // 256}.{cat_idx % 256 + 1}"
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"X-Forwarded-For": fake_ip},
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        if headed and start_delay_sec > 0:
            print(f"  [{industry}/{category}] Waiting {start_delay_sec}s before starting (watch mode)...")
            page.wait_for_timeout(start_delay_sec * 1000)

        try:
            def _click_category_chip(target_category: str, timeout_ms: int = 12000) -> bool:
                """Click a visible category chip (no JS state mutation fallbacks)."""
                target_norm = re.sub(r"[^a-z0-9]+", "", target_category.lower())
                deadline = time.time() + (timeout_ms / 1000.0)
                while time.time() < deadline:
                    chips = page.locator("#suggestedCategories .suggest-chip")
                    count = chips.count()
                    if count > 0:
                        # Exact normalized match first.
                        for i in range(count):
                            text = (chips.nth(i).text_content() or "").strip().lower()
                            norm = re.sub(r"[^a-z0-9]+", "", text.replace(" ", "_"))
                            if norm == target_norm:
                                chips.nth(i).click()
                                return True
                        # Fuzzy containment fallback.
                        for i in range(count):
                            text = (chips.nth(i).text_content() or "").strip().lower()
                            norm = re.sub(r"[^a-z0-9]+", "", text)
                            if target_norm in norm or norm in target_norm:
                                chips.nth(i).click()
                                return True
                        return False
                    page.wait_for_timeout(250)
                return False

            def _extract_first_number(text: str) -> float | None:
                if not text:
                    return None
                m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
                if not m:
                    return None
                try:
                    return float(m.group(0))
                except Exception:
                    return None

            def _fill_category_questions() -> int:
                """Fill dynamic Section B+ fields shown for selected category."""
                section = page.locator("#categoryQuestionsSection")
                if section.count() == 0:
                    return 0
                try:
                    if not section.first.is_visible():
                        return 0
                except Exception:
                    return 0

                fields = page.locator("#categoryQuestionsContainer input, #categoryQuestionsContainer textarea, #categoryQuestionsContainer select")
                count = fields.count()
                if count == 0:
                    return 0

                filled = 0
                for i in range(count):
                    el = fields.nth(i)
                    try:
                        is_rendered = bool(
                            el.evaluate(
                                "(e) => Boolean(e && (e.offsetParent || e.getClientRects().length))"
                            )
                        )
                        if not is_rendered:
                            continue
                        el.scroll_into_view_if_needed()
                    except Exception:
                        continue

                    try:
                        meta = el.evaluate(
                            """(e) => {
                                const group = e.closest('.form-group');
                                const label = group ? (group.querySelector('label')?.innerText || '') : '';
                                const options = e.tagName.toLowerCase() === 'select'
                                  ? Array.from(e.options).map(o => ({ value: String(o.value || ''), text: String(o.textContent || '') }))
                                  : [];
                                return {
                                    tag: e.tagName.toLowerCase(),
                                    type: String(e.getAttribute('type') || ''),
                                    name: String(e.getAttribute('name') || ''),
                                    placeholder: String(e.getAttribute('placeholder') || ''),
                                    min: String(e.getAttribute('min') || ''),
                                    max: String(e.getAttribute('max') || ''),
                                    step: String(e.getAttribute('step') || ''),
                                    disabled: Boolean(e.disabled),
                                    label: String(label || ''),
                                    options
                                };
                            }"""
                        )
                    except Exception:
                        continue

                    if meta.get("disabled"):
                        continue

                    tag = str(meta.get("tag") or "").lower()
                    input_type = str(meta.get("type") or "").lower()
                    name = str(meta.get("name") or "")
                    placeholder = str(meta.get("placeholder") or "")
                    label = str(meta.get("label") or "")
                    prompt = f"{label} {placeholder} {name}".strip()

                    if tag == "select":
                        options = [o for o in (meta.get("options") or []) if str((o or {}).get("value") or "").strip()]
                        if not options:
                            continue
                        answer_seed = _pick_answer(name or placeholder, prompt, industry).lower()
                        tokens = [t for t in re.findall(r"[a-z0-9]+", answer_seed) if len(t) >= 4]
                        choice = None
                        for opt in options:
                            blob = f"{opt.get('value', '')} {opt.get('text', '')}".lower()
                            if any(tok in blob for tok in tokens):
                                choice = opt
                                break
                        if choice is None:
                            choice = options[0]
                        el.select_option(str(choice.get("value") or ""))
                        filled += 1
                        continue

                    if tag == "textarea" or input_type in {"text", "search", "email", "url", ""}:
                        answer = _pick_answer(name or placeholder, prompt, industry)
                        if not answer:
                            answer = f"{cat_label} baseline remains stable with periodic spikes."
                        el.fill(answer)
                        filled += 1
                        continue

                    if input_type == "number":
                        seed = _pick_answer(name or placeholder, prompt, industry)
                        num = _extract_first_number(seed)
                        if num is None:
                            num = 50.0
                        low_prompt = prompt.lower()
                        if "percent" in low_prompt and num > 100:
                            num = 85.0

                        min_raw = str(meta.get("min") or "").strip()
                        max_raw = str(meta.get("max") or "").strip()
                        try:
                            if min_raw:
                                num = max(num, float(min_raw))
                        except Exception:
                            pass
                        try:
                            if max_raw:
                                num = min(num, float(max_raw))
                        except Exception:
                            pass

                        if any(tok in low_prompt for tok in ("count", "days", "orders", "id", "number", "volume", "qty", "quantity")):
                            num = float(int(round(num)))

                        out = str(int(num)) if float(num).is_integer() else f"{num:.2f}"
                        el.fill(out)
                        filled += 1
                        continue
                    # Any unsupported type: leave untouched.
                return filled

            def _attempt_open_toji_from_error() -> tuple[bool, str | None]:
                """Recover from insufficient-match synthesis errors by opening Toji intake."""
                nonlocal page
                error_box = page.locator("#errorMessage")
                error_text = ""
                if error_box.count() > 0:
                    try:
                        error_text = (error_box.text_content() or "").strip()
                    except Exception:
                        error_text = ""
                if not error_text:
                    return False, None

                lowered = error_text.lower()
                is_match_gate = (
                    "matched" in lowered and "need at least" in lowered
                ) or ("answer a few questions in toji" in lowered)
                if not is_match_gate:
                    return False, error_text

                link = page.locator('#errorMessage a[href*="chat.html"]')
                popup = None
                fallback_chat_href = None
                if link.count() > 0:
                    try:
                        fallback_chat_href = link.first.get_attribute("href")
                    except Exception:
                        fallback_chat_href = None
                    try:
                        with page.expect_popup(timeout=12000) as popup_info:
                            link.first.click()
                        popup = popup_info.value
                    except Exception:
                        popup = None

                if popup is None:
                    # Popup may be blocked in some browser contexts; open intake directly.
                    if fallback_chat_href and "chat.html" in fallback_chat_href:
                        target = fallback_chat_href if fallback_chat_href.startswith("http") else f"{base_url}/{fallback_chat_href.lstrip('./')}"
                    else:
                        target = f"{base_url}/chat.html?intake_mode=1&industry={industry}&category={category}"
                    page.goto(target)
                    return True, error_text

                page = popup
                page.set_default_timeout(timeout_ms)
                page.wait_for_load_state("networkidle")
                return True, error_text

            def _wait_for_post_submit_redirect(max_wait_ms: int = 120000) -> str | None:
                """Detect redirect in same tab or delayed popup tab."""
                nonlocal page
                deadline = time.time() + (max_wait_ms / 1000.0)
                while time.time() < deadline:
                    try:
                        current_url = page.url or ""
                    except Exception:
                        current_url = ""
                    if "chat.html" in current_url:
                        return "chat"
                    if "dashboard.html" in current_url:
                        return "dashboard"

                    # Some flows open chat/dashboard in a new popup after async calls.
                    for candidate in list(context.pages):
                        if candidate == page:
                            continue
                        try:
                            url = candidate.url or ""
                        except Exception:
                            continue
                        if "chat.html" in url or "dashboard.html" in url:
                            page = candidate
                            page.set_default_timeout(timeout_ms)
                            return "chat" if "chat.html" in url else "dashboard"

                    page.wait_for_timeout(400)
                return None

            # ── Step 1: Tell Us form ──────────────────────────────
            print(f"  [{industry}/{category}] Step 1: Filling Tell Us form...")
            page.goto(f"{base_url}/synthesis.html")
            page.wait_for_load_state("networkidle")

            # Select industry
            industry_btn = page.locator(f'.industry-option[data-industry="{industry}"]')
            if industry_btn.count() == 0:
                result["errors"].append(f"Industry button not found for '{industry}'")
                result["status"] = "stuck_at_tellus"
                result["duration_sec"] = time.time() - t0
                browser.close()
                return result
            industry_btn.first.scroll_into_view_if_needed()
            industry_btn.first.click()
            page.wait_for_timeout(800)

            # Wait for category chips to load, then select our target category
            page.wait_for_timeout(1500)
            if not _click_category_chip(category):
                result["errors"].append(f"Category chip not clickable for '{category}'")
                result["status"] = "stuck_at_tellus"
                result["duration_sec"] = time.time() - t0
                browser.close()
                return result
            page.wait_for_timeout(300)

            # Fill 9 form fields
            for qid in ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"]:
                el = page.locator(f"#{qid}")
                if el.count() == 0:
                    continue
                tag = el.evaluate("e => e.tagName.toLowerCase()")
                val = persona[qid]
                if tag == "select":
                    # Pick the option closest to our persona value
                    options = el.evaluate("e => Array.from(e.options).map(o => o.value)")
                    non_empty = [str(opt) for opt in (options or []) if str(opt).strip()]
                    if non_empty:
                        # Try to find a match, otherwise pick first non-empty.
                        best = non_empty[0]
                        for opt in non_empty:
                            if any(word in opt.lower() for word in val.lower().split()):
                                best = opt
                                break
                        el.select_option(best)
                elif tag == "textarea":
                    el.fill(val)
                else:
                    el.fill(val)

            # Fill entity name
            entity_el = page.locator("#entity_name")
            if entity_el.count() > 0:
                entity_el.fill(persona.get("entity_name", "Test Corp"))

            filled_dynamic = _fill_category_questions()
            result["category_fields_filled"] = int(filled_dynamic)
            if filled_dynamic > 0:
                print(f"  [{industry}/{category}] Step 1b: Filled {filled_dynamic} category detail field(s)")

            result["tell_us"] = "filled"

            # Take screenshot before submit
            ss_dir = screenshot_dir / industry / category
            ss_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(ss_dir / "01_tell_us_filled.png"))

            # Ensure all required form fields are valid before submit.
            invalid_fields = page.locator("#synthesisForm :invalid")
            invalid_count = invalid_fields.count()
            if invalid_count > 0:
                bad = []
                for i in range(min(8, invalid_count)):
                    try:
                        meta = invalid_fields.nth(i).evaluate(
                            """(e) => ({
                                id: String(e.id || ''),
                                name: String(e.name || ''),
                                placeholder: String(e.placeholder || ''),
                                tag: String(e.tagName || '').toLowerCase()
                            })"""
                        )
                        bad.append(
                            meta.get("name")
                            or meta.get("id")
                            or meta.get("placeholder")
                            or meta.get("tag")
                            or "unknown"
                        )
                    except Exception:
                        bad.append("unknown")
                page.screenshot(path=str(ss_dir / "01_invalid_fields.png"))
                result["errors"].append(
                    f"Form has {invalid_count} invalid required field(s) before submit: {', '.join(bad)}"
                )
                result["status"] = "stuck_at_tellus"
                result["duration_sec"] = time.time() - t0
                browser.close()
                return result

            # Submit the form. Current UX may open chat intake in a popup tab.
            submit_btn = page.locator("#submitBtn:visible")
            if submit_btn.count() == 0:
                submit_btn = page.locator("#submitBtn")
            popup_page = None
            try:
                submit_btn.first.scroll_into_view_if_needed()
            except Exception:
                pass
            try:
                submit_btn.first.click(timeout=15000, force=True)
            except Exception as exc:
                page.screenshot(path=str(ss_dir / "01_submit_click_failed.png"))
                result["errors"].append(f"Submit click failed: {type(exc).__name__}: {exc}")
                result["status"] = "stuck_at_tellus"
                result["duration_sec"] = time.time() - t0
                browser.close()
                return result
            try:
                popup_page = context.wait_for_event("page", timeout=5000)
            except Exception:
                popup_page = None

            if popup_page is not None:
                page = popup_page
                page.set_default_timeout(timeout_ms)

            print(f"  [{industry}/{category}] Step 1: Form submitted, waiting for redirect...")

            redirected_to = _wait_for_post_submit_redirect(
                max_wait_ms=max(420000, int(timeout_ms * 3))
            )
            if not redirected_to:
                # Maybe still on synthesis/chat page with an error state
                page.screenshot(path=str(ss_dir / "01b_stuck.png"))
                err_el = page.locator("#errorMessage")
                err_text = err_el.text_content() if err_el.count() > 0 and err_el.is_visible() else ""
                recovered, recovered_error = _attempt_open_toji_from_error()
                if recovered:
                    redirected_to = "chat"
                    if recovered_error:
                        result["errors"].append(
                            f"Recovered from insufficient-match gate via Toji intake: {recovered_error[:240]}"
                        )
                else:
                    result["errors"].append(f"No redirect after submit. Error: {err_text}")
                    result["status"] = "stuck_at_tellus"
                    result["duration_sec"] = time.time() - t0
                    browser.close()
                    return result

            page.screenshot(path=str(ss_dir / "02_after_redirect.png"))

            # ── Step 2: Toji intake conversation ──────────────────
            if redirected_to == "chat":
                print(f"  [{industry}/{category}] Step 2: In Toji intake mode, answering questions...")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(3000)  # Let Toji ask first question

                max_questions = 30
                question_count = 0
                stuck_count = 0
                last_msg_count = 0

                for q_round in range(max_questions):
                    # Wait for Toji to finish typing
                    page.wait_for_timeout(2000)

                    # Count messages
                    messages = page.locator("#messages .message")
                    msg_count = messages.count()

                    if msg_count == last_msg_count:
                        stuck_count += 1
                        if stuck_count > 3:
                            break
                    else:
                        stuck_count = 0
                    last_msg_count = msg_count

                    # Check if we got redirected to dashboard (pipeline completed)
                    if "dashboard.html" in page.url:
                        break

                    # Check for completion messages
                    page_text = page.locator("#messages").text_content() or ""
                    if "Taking you to the dashboard" in page_text or "analysis is ready" in page_text.lower():
                        page.wait_for_timeout(3000)
                        break
                    if "All done" in page_text or "good to go" in page_text.lower():
                        break

                    # Check if input is enabled (Toji is waiting for answer)
                    input_el = page.locator("#input")
                    if not input_el.is_enabled():
                        page.wait_for_timeout(1000)
                        continue

                    # Get the last assistant message to understand what Toji asked
                    assistant_msgs = page.locator("#messages .message.assistant")
                    if assistant_msgs.count() == 0:
                        continue
                    last_assistant = assistant_msgs.last.text_content() or ""

                    # Detect Toji reframes: "rephrase", "approximate", "skip", "ballpark"
                    is_reframe = any(w in last_assistant.lower() for w in (
                        "rephrase", "approximate", "ballpark", "skip", "simplify",
                        "rough estimate", "move on", "safe default", "no pressure",
                        "let me try a different", "i appreciate the response",
                    ))

                    if is_reframe:
                        # Toji is reframing — give "skip" or a simple fallback
                        input_el.fill("skip")
                        page.locator("#sendBtn").click()
                        page.wait_for_timeout(2000)
                        continue

                    # If Toji asked a question, extract field context and answer
                    if "?" in last_assistant or "[" in last_assistant[:5]:
                        question_count += 1
                        result["intake_questions_asked"].append(last_assistant[:200])

                        # Determine what kind of answer to give
                        answer = _pick_answer(
                            field_name=last_assistant[:60],
                            question_text=last_assistant,
                            industry=industry,
                        )

                        # Every 5th question, ask a counter-question to test reframing
                        if question_count % 5 == 0 and question_count > 0:
                            counter = "What do you mean by that exactly? Can you clarify?"
                            input_el.fill(counter)
                            page.locator("#sendBtn").click()
                            page.wait_for_timeout(4000)

                            # Get reframe response
                            reframe_msg = page.locator("#messages .message.assistant").last.text_content() or ""
                            result["toji_reframes"].append({
                                "question": last_assistant[:150],
                                "counter": counter,
                                "reframe": reframe_msg[:300],
                            })

                            # Now give the real answer
                            page.wait_for_timeout(500)

                        input_el.fill(answer)
                        result["intake_answers_given"].append(answer[:100])
                        page.locator("#sendBtn").click()
                        page.wait_for_timeout(1500)
                    else:
                        # Not a question — maybe a processing message
                        page.wait_for_timeout(2000)

                result["intake_questions_count"] = question_count
                page.screenshot(path=str(ss_dir / "03_intake_done.png"))
                print(f"  [{industry}/{category}] Step 2: Answered {question_count} questions")

                # Wait for redirect to dashboard
                try:
                    redirect_timeout = max(90000, int(timeout_ms * 2.5))
                    page.wait_for_url("**/dashboard.html**", timeout=redirect_timeout)
                except Exception:
                    # Maybe pipeline is still running or we're still in chat
                    page.screenshot(path=str(ss_dir / "03b_no_dashboard_redirect.png"))
                    if "dashboard.html" not in page.url:
                        result["errors"].append("Did not redirect to dashboard after intake")

            # ── Step 3: Dashboard / Report ────────────────────────
            if "dashboard.html" in page.url:
                print(f"  [{industry}/{category}] Step 3: On dashboard, extracting report...")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(3000)
                page.screenshot(path=str(ss_dir / "04_dashboard.png"))

                # Extract report_id from URL
                url_params = page.url.split("?")[-1] if "?" in page.url else ""
                report_match = re.search(r"report_id=([^&]+)", url_params)
                if report_match:
                    result["report_id"] = report_match.group(1)

                # ── Step 4: Navigate to chat with report context ──
                # Pause to let Groq rate limits recover after the pipeline's LLM calls
                print(f"  [{industry}/{category}] Step 4: Opening Toji chat with report (waiting for rate limit cooldown)...")
                page.wait_for_timeout(15000)
                report_id_param = f"report_id={result['report_id']}" if result["report_id"] else ""
                page.goto(f"{base_url}/chat.html?{report_id_param}")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(8000)  # Wait for briefing to load

                page.screenshot(path=str(ss_dir / "05_toji_briefing.png"))

                # Get the briefing message
                assistant_msgs = page.locator("#messages .message.assistant")
                if assistant_msgs.count() > 0:
                    briefing_text = assistant_msgs.first.text_content() or ""
                    result["briefing"] = briefing_text[:500]
                    print(f"  [{industry}/{category}] Step 4: Briefing received ({len(briefing_text)} chars)")

                # ── Step 5: Ask follow-up questions ───────────────
                followups = EXEC_FOLLOWUPS.get(industry, [
                    "What are the top 3 things I should focus on based on this report?",
                    "How confident are you in these recommendations given the data quality?",
                ])
                for i, question in enumerate(followups):
                    print(f"  [{industry}/{category}] Step 5: Asking follow-up {i+1}...")
                    # Brief pause between questions to avoid rate limits
                    if i > 0:
                        page.wait_for_timeout(5000)
                    input_el = page.locator("#input")
                    input_el.fill(question)
                    page.locator("#sendBtn").click()

                    # Wait for response
                    page.wait_for_timeout(10000)

                    # Get last assistant message
                    assistant_msgs = page.locator("#messages .message.assistant")
                    if assistant_msgs.count() > 0:
                        response = assistant_msgs.last.text_content() or ""
                        result["followup_exchanges"].append({
                            "question": question,
                            "response": response[:500],
                        })
                    page.screenshot(path=str(ss_dir / f"06_followup_{i+1}.png"))

            result["status"] = "completed"

        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            result["status"] = "error"
            traceback.print_exc()
            try:
                page.screenshot(path=str(ss_dir / "99_error.png"))
            except Exception:
                pass

        result["duration_sec"] = round(time.time() - t0, 1)
        if headed and hold_open_sec > 0:
            print(f"  [{industry}/{category}] Holding browser open for {hold_open_sec}s...")
            try:
                page.wait_for_timeout(hold_open_sec * 1000)
            except Exception:
                time.sleep(max(1, hold_open_sec))
        browser.close()

    return result


# ---------------------------------------------------------------------------
# All 37 testable categories
# ---------------------------------------------------------------------------

ALL_CATEGORIES: dict[str, list[str]] = {
    "ecommerce": [
        "basket_intelligence", "catalog_quality", "checkout_risk",
        "conversion_optimization", "demand_signal", "fulfillment_flow", "merchandising",
    ],
    "shipping_freight": [
        "carrier_safety_risk", "claims_damage_risk", "cold_chain_integrity",
        "cross_border_flow", "customs_compliance", "eta_delay_risk",
        "fleet_utilization", "inland_waterway_flow", "lane_cost_yield",
        "last_mile_sla", "ocean_schedule_reliability", "port_terminal_congestion",
        "rail_intermodal_flow", "trade_volume_mix", "trucking_capacity",
    ],
    "trucking_delivery": [
        "cold_chain_last_mile", "cross_border_trucking", "dispatch_capacity_balance",
        "driver_safety_compliance", "fleet_utilization", "fuel_energy_efficiency",
        "lane_cost_yield", "last_mile_sla", "parcel_exception_risk",
        "pickup_dropoff_reliability", "reverse_logistics_returns", "route_eta_reliability",
        "urban_traffic_risk", "vehicle_maintenance_risk", "workforce_shift_planning",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="E2E Toji test across all categories")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--industry", default=None, help="Test only this industry")
    parser.add_argument("--category", default=None, help="Test only this category")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--timeout", type=int, default=120000, help="Per-action timeout ms")
    parser.add_argument("--start-delay", type=int, default=0, help="Seconds to wait before starting actions in headed mode")
    parser.add_argument("--hold-open", type=int, default=0, help="Seconds to keep browser open after scenario in headed mode")
    parser.add_argument("--output", default="/tmp/toji_e2e_results.json", help="Results output path")
    args = parser.parse_args()

    screenshot_dir = Path("/tmp/toji_screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Build test matrix
    test_matrix = []
    for ind, cats in ALL_CATEGORIES.items():
        if args.industry and ind != args.industry:
            continue
        for cat in cats:
            if args.category and cat != args.category:
                continue
            test_matrix.append((ind, cat))

    print(f"\n{'='*60}")
    print(f"TOJI E2E TEST — {len(test_matrix)} categories")
    print(f"{'='*60}\n")

    results = []
    for idx, (industry, category) in enumerate(test_matrix):
        print(f"\n[{idx+1}/{len(test_matrix)}] {industry} / {category}")
        print("-" * 50)

        r = run_single_category(
            industry=industry,
            category=category,
            base_url=args.base_url,
            headed=args.headed,
            timeout_ms=args.timeout,
            screenshot_dir=screenshot_dir,
            start_delay_sec=max(0, int(args.start_delay)),
            hold_open_sec=max(0, int(args.hold_open)),
        )
        results.append(r)

        status_icon = "✓" if r["status"] == "completed" else "✗"
        print(f"  {status_icon} Status: {r['status']} | Questions: {r['intake_questions_count']} | "
              f"Reframes: {len(r['toji_reframes'])} | Duration: {r['duration_sec']}s")
        if r["errors"]:
            for e in r["errors"]:
                print(f"    ERROR: {e[:120]}")
        if r["briefing"]:
            print(f"    Briefing preview: {r['briefing'][:120]}...")

        # Write incremental results
        Path(args.output).write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    completed = sum(1 for r in results if r["status"] == "completed")
    errored = sum(1 for r in results if r["status"] == "error")
    stuck = sum(1 for r in results if r["status"] not in ("completed", "error"))
    print(f"  Completed: {completed}/{len(results)}")
    print(f"  Errored:   {errored}/{len(results)}")
    print(f"  Stuck:     {stuck}/{len(results)}")
    print(f"  Total questions asked across all: {sum(r['intake_questions_count'] for r in results)}")
    print(f"  Total reframes tested: {sum(len(r['toji_reframes']) for r in results)}")
    print(f"  Screenshots: {screenshot_dir}")
    print(f"  Full results: {args.output}")


if __name__ == "__main__":
    main()
