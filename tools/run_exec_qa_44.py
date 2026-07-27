#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.app import create_app  # noqa: E402


@dataclass
class CategoryRun:
    industry: str
    category: str
    session_id: str
    task_id: str | None
    report_id: str | None
    success: bool
    task_state: str
    match_score: float | None
    risk_score: float | None
    rows: int | None
    missing_fields: int
    visuals_count: int
    has_trend: bool
    has_forecast: bool
    guided_count: int
    guided_has_time_anchor: bool
    guided_all_have_reframe: bool
    toji_answers_ok: bool
    toji_offtopic_guard_ok: bool
    issues: list[str]


def load_categories() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted((ROOT / "config" / "router" / "manifests").glob("*_router_manifest.json")):
        payload = json.loads(path.read_text())
        industry = str(payload.get("industry") or path.name.replace("_router_manifest.json", "")).strip()
        for row in payload.get("categories") or []:
            cat = str((row or {}).get("category") or "").strip()
            if cat:
                out.append((industry, cat))
    return out


def fmt_category(cat: str) -> str:
    return cat.replace("_", " ")


def answer_from_question(industry: str, category: str, q: str, idx: int) -> str:
    text = q.lower()
    if "how long" in text or "since" in text or "time window" in text:
        return "This has been happening for the past 4 months."
    if "season" in text or "seasonality" in text:
        return "7"
    if "region" in text or "geograph" in text or "market" in text:
        if industry == "ecommerce":
            return "United States and Western Europe."
        if industry == "shipping_freight":
            return "US West Coast, Northern Europe, and East Asia corridors."
        if industry == "trucking_delivery":
            return "US metro hubs and regional suburban routes."
        return "North America and Europe."
    if "goal" in text or "objective" in text:
        return f"Reduce operational risk and improve {fmt_category(category)} performance over the next quarter."
    if "kpi" in text or "target" in text:
        return "Increase reliability by 12 percent while reducing avoidable cost by 8 percent."
    if "constraint" in text or "limit" in text:
        return "Budget and staffing are constrained, so we need low-disruption interventions first."
    if "volume" in text or "demand" in text:
        return "Demand is volatile with weekly spikes; baseline is stable with occasional surges."
    if "risk" in text:
        return "Current risk is medium-high with concentration around a few operational bottlenecks."
    if "cost" in text or "price" in text:
        return "Unit economics are under pressure from variable transport and handling costs."
    if "sla" in text or "delay" in text:
        return "SLA misses increased in the last 2 months, especially during high-volume windows."
    if "conversion" in text:
        return "Conversion softened during peak campaigns and improved after checkout simplification."
    if "inventory" in text or "stock" in text:
        return "Stockouts are concentrated in fast-moving SKUs and late replenishment windows."
    defaults = [
        "This issue is measurable and has become operationally material in the last quarter.",
        "We can provide rough values and trend direction even if exact values are unavailable.",
        "The biggest problem is volatility and uneven performance across lanes/segments.",
    ]
    return defaults[idx % len(defaults)]


def build_context(industry: str, category: str, question_payload: dict[str, Any], exec_idx: int) -> tuple[str, dict[str, Any]]:
    iq = [str(x) for x in (question_payload.get("industry_questions") or [])]
    cq = [str(x) for x in (question_payload.get("category_questions") or [])]
    csets = question_payload.get("category_question_sets") or []

    if len(iq) < 3:
        iq = [
            "What is the primary business objective?",
            "What operating region is most important?",
            "What constraints should be respected?",
        ]
    if len(cq) < 3:
        cq = [
            f"What is the core issue in {fmt_category(category)}?",
            "What KPI is underperforming?",
            "What target outcome is needed?",
        ]

    industry_answers = [answer_from_question(industry, category, q, i) for i, q in enumerate(iq[:3])]
    category_answers = [answer_from_question(industry, category, q, i + 10) for i, q in enumerate(cq[:3])]

    answers_by_cat: dict[str, list[str]] = {}
    for row in csets:
        if not isinstance(row, dict):
            continue
        c = str(row.get("category") or "").strip()
        qs = [str(x) for x in (row.get("questions") or [])]
        if not c:
            continue
        answers_by_cat[c] = [answer_from_question(industry, c, q, i + 20) for i, q in enumerate(qs)]

    if category not in answers_by_cat:
        answers_by_cat[category] = [answer_from_question(industry, category, q, i + 30) for i, q in enumerate(cq)]

    problem_statement = (
        f"As COO of {industry} operations, I need to improve {fmt_category(category)} outcomes. "
        "We need a practical plan that improves reliability, reduces risk, and protects margin in the next 30 days."
    )

    now = datetime.now(timezone.utc)
    start = now.replace(day=1)
    start_iso = start.isoformat()

    context = {
        "entity_name": f"ExecSim-{industry}-{exec_idx:02d}",
        "category": category,
        "q1": industry_answers[0],
        "q2": industry_answers[1],
        "q3": industry_answers[2],
        "q4": category_answers[0],
        "q5": category_answers[1],
        "q6": category_answers[2],
        "q7": "past 4 months",
        "q8": "7",
        "q9": "Improve leading KPI by 10-15% with no SLA regression.",
        "industry_answers": industry_answers,
        "category_answers": category_answers,
        "category_answers_by_category": answers_by_cat,
        "workflow_goal": f"Stabilize and improve {fmt_category(category)} performance.",
        "kpi_baseline_target": "10-15% improvement in primary KPI",
        "constraints": "Budget and staffing constraints, minimize operational disruption.",
        "time_context_answer": "past 4 months",
        "issue_duration_text": "past 4 months",
        "lookback_days": 120,
        "analysis_start_date": start_iso,
        "analysis_end_date": now.isoformat(),
    }
    return problem_statement, context


def poll_task(client: TestClient, task_id: str, headers: dict[str, str], timeout_sec: int = 420) -> tuple[str, dict[str, Any]]:
    start = time.time()
    last_payload: dict[str, Any] = {}
    while True:
        r = client.get(f"/task/{task_id}", headers=headers)
        payload = r.json()
        last_payload = payload
        state = str(payload.get("state") or "")
        if state in {"REPORT_READY", "ERROR"}:
            return state, payload
        if time.time() - start > timeout_sec:
            return "TIMEOUT", payload
        time.sleep(1.2)


def validate_toji_response(text: str) -> tuple[bool, bool]:
    t = (text or "").strip()
    if not t:
        return False, False
    on_course = True
    for bad in ["internal prompt", "system prompt", "developer message", "hidden instruction"]:
        if bad in t.lower():
            on_course = False
            break
    provider_safe = True
    for leak in ["groq", "gpt-oss", "llama", "openai model", "provider is"]:
        if re.search(rf"\b{re.escape(leak)}\b", t.lower()):
            provider_safe = False
            break
    return on_course, provider_safe


def run_one(client: TestClient, industry: str, category: str, exec_idx: int) -> CategoryRun:
    issues: list[str] = []
    session_id = f"exec_{industry}_{category}_{exec_idx}"
    headers = {
        "user-agent": f"ExecSim/{industry}/{category}/{exec_idx}",
        "x-forwarded-for": f"10.0.{(exec_idx % 200) + 1}.{(exec_idx % 240) + 1}",
    }

    task_id = None
    report_id = None
    report: dict[str, Any] = {}

    # Start session
    r = client.post("/session/start", headers=headers)
    if r.status_code not in {200, 201}:
        return CategoryRun(industry, category, session_id, None, None, False, "SESSION_FAIL", None, None, None, 0, 0, False, False, 0, False, False, False, False, [f"session_start {r.status_code}"])

    # Ask question set
    qres = client.post("/intake/tell-us/questions", headers=headers, json={"industry": industry, "category": category})
    if qres.status_code != 200:
        issues.append(f"questions_failed:{qres.status_code}")
        qpayload: dict[str, Any] = {}
    else:
        qpayload = qres.json()

    problem_statement, context = build_context(industry, category, qpayload, exec_idx)

    # Tell us
    tres = client.post(
        "/intake/tell-us",
        headers=headers,
        json={"problem_statement": problem_statement, "industry": industry, "context": context},
    )
    if tres.status_code != 200:
        return CategoryRun(industry, category, session_id, None, None, False, "TELL_US_FAIL", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + [f"tell_us {tres.status_code} {tres.text[:200]}"])

    # Confirm synth
    cres = client.post("/intake/tell-us/confirm", headers=headers, json={"industry": industry, "use_synthetic": True})
    if cres.status_code != 200:
        return CategoryRun(industry, category, session_id, None, None, False, "CONFIRM_FAIL", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + [f"confirm {cres.status_code} {cres.text[:200]}"])
    artifact_uri = (cres.json() or {}).get("artifact_uri")
    if not artifact_uri:
        return CategoryRun(industry, category, session_id, None, None, False, "CONFIRM_NO_ARTIFACT", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + ["confirm returned no artifact_uri"])

    # Predict
    pres = client.post("/run/predict", headers=headers, json={"industry": industry, "category": category, "artifact_uri": artifact_uri})
    if pres.status_code != 202:
        return CategoryRun(industry, category, session_id, None, None, False, "PREDICT_FAIL", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + [f"predict {pres.status_code} {pres.text[:200]}"])
    task_id = (pres.json() or {}).get("task_id")
    if not task_id:
        return CategoryRun(industry, category, session_id, None, None, False, "NO_TASK_ID", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + ["predict returned no task_id"])

    state, task_payload = poll_task(client, task_id, headers=headers)
    if state != "REPORT_READY":
        issues.append(f"task_state:{state}")
        return CategoryRun(industry, category, session_id, task_id, None, False, state, None, None, None, 0, 0, False, False, 0, False, False, False, False, issues + [str(task_payload)[:500]])

    report_id = str(task_payload.get("report_id") or "").strip()
    if not report_id:
        issues.append("task_ready_no_report_id")
        return CategoryRun(industry, category, session_id, task_id, None, False, state, None, None, None, 0, 0, False, False, 0, False, False, False, False, issues)

    rrep = client.get(f"/report/{report_id}", headers=headers)
    if rrep.status_code != 200:
        issues.append(f"report_get:{rrep.status_code}")
        return CategoryRun(industry, category, session_id, task_id, report_id, False, "REPORT_FETCH_FAIL", None, None, None, 0, 0, False, False, 0, False, False, False, False, issues)
    report = rrep.json() or {}

    scorecard = report.get("scorecard") or {}
    routing = report.get("routing") or {}
    visuals = report.get("visuals") or []
    visual_names = {str(v.get("name") or "") for v in visuals if isinstance(v, dict)}
    has_trend = "trend" in visual_names
    has_forecast = "forecast" in visual_names

    if report.get("visuals_error"):
        issues.append(f"visuals_error:{report.get('visuals_error')}")

    # Guided questions if missing fields
    missing = list(routing.get("missing_fields") or [])
    guided_count = 0
    guided_time = False
    guided_reframe = False
    if missing:
        g = client.post(
            "/chat/guided-questions",
            headers=headers,
            json={"report_id": report_id, "missing_fields": missing, "max_questions": 15},
        )
        if g.status_code == 200:
            gp = g.json() or {}
            qs = gp.get("questions") or []
            guided_count = len(qs)
            if qs:
                first_q = str((qs[0] or {}).get("question") or "").lower()
                guided_time = "how long has this been happening" in first_q
                guided_reframe = all(bool(str((row or {}).get("reframe") or "").strip()) for row in qs)
            else:
                issues.append("guided_questions_empty")
        else:
            issues.append(f"guided_questions_fail:{g.status_code}")

    # Chat with Toji based on dashboard report
    prompts = [
        "What should we do in the next 30 days to improve outcomes?",
        "What additional data should we collect to improve confidence?",
        "Are you gpt or groq?",
    ]
    toji_ok = True
    offtopic_ok = True
    for i, msg in enumerate(prompts):
        cm = client.post("/chat/message", headers=headers, json={"report_id": report_id, "message": msg})
        if cm.status_code != 200:
            issues.append(f"chat_fail_{i}:{cm.status_code}")
            toji_ok = False
            continue
        payload = cm.json() or {}
        resp_text = str(payload.get("response") or "")
        on_course, provider_safe = validate_toji_response(resp_text)
        toji_ok = toji_ok and on_course
        if i == 2:
            offtopic_ok = provider_safe and ("industry" in resp_text.lower() or "operational" in resp_text.lower() or "analysis" in resp_text.lower())

    success = not issues and has_forecast and bool(report_id)

    return CategoryRun(
        industry=industry,
        category=category,
        session_id=session_id,
        task_id=task_id,
        report_id=report_id,
        success=success,
        task_state=state,
        match_score=(float(scorecard.get("match_score")) if scorecard.get("match_score") is not None else None),
        risk_score=(float(scorecard.get("risk_score")) if scorecard.get("risk_score") is not None else None),
        rows=(int(scorecard.get("rows")) if scorecard.get("rows") is not None else None),
        missing_fields=len(missing),
        visuals_count=len(visuals),
        has_trend=has_trend,
        has_forecast=has_forecast,
        guided_count=guided_count,
        guided_has_time_anchor=guided_time,
        guided_all_have_reframe=guided_reframe,
        toji_answers_ok=toji_ok,
        toji_offtopic_guard_ok=offtopic_ok,
        issues=issues,
    )


def write_reports(results: list[CategoryRun], started_at: datetime) -> tuple[Path, Path]:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    out_json = reports_dir / f"exec_qa_44_{stamp}.json"
    out_md = reports_dir / f"exec_qa_44_{stamp}.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "success_count": sum(1 for r in results if r.success),
        "with_issues": sum(1 for r in results if r.issues),
        "rows": [asdict(r) for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines: list[str] = []
    lines.append("# Executive QA Simulation (44 Categories)\n")
    lines.append(f"Generated: `{payload['generated_at']}`\n")
    lines.append(f"Total categories: **{payload['total']}**")
    lines.append(f"Success: **{payload['success_count']}**")
    lines.append(f"With issues: **{payload['with_issues']}**\n")

    lines.append("| Industry | Category | Task | Match | Risk | Missing | Visuals | Guided Q | Time Q | Reframes | Toji | Guard | Issues |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for r in results:
        issues = "; ".join(r.issues[:2]) if r.issues else ""
        lines.append(
            f"| {r.industry} | {r.category} | {r.task_state} | "
            f"{'' if r.match_score is None else f'{r.match_score:.4f}'} | "
            f"{'' if r.risk_score is None else f'{r.risk_score:.4f}'} | "
            f"{r.missing_fields} | {r.visuals_count} | {r.guided_count} | "
            f"{r.guided_has_time_anchor} | {r.guided_all_have_reframe} | "
            f"{r.toji_answers_ok} | {r.toji_offtopic_guard_ok} | {issues} |"
        )

    by_industry: dict[str, list[CategoryRun]] = {}
    for r in results:
        by_industry.setdefault(r.industry, []).append(r)

    lines.append("\n## Industry Summary\n")
    lines.append("| Industry | Categories | Success | Avg Match | Avg Risk | Avg Missing | Forecast Coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ind, rows in sorted(by_industry.items()):
        succ = sum(1 for x in rows if x.success)
        mm = [x.match_score for x in rows if x.match_score is not None]
        rr = [x.risk_score for x in rows if x.risk_score is not None]
        miss = [x.missing_fields for x in rows]
        fc = sum(1 for x in rows if x.has_forecast)
        avg_match = sum(mm) / len(mm) if mm else 0.0
        avg_risk = sum(rr) / len(rr) if rr else 0.0
        avg_missing = sum(miss) / len(miss) if miss else 0.0
        lines.append(f"| {ind} | {len(rows)} | {succ} | {avg_match:.4f} | {avg_risk:.4f} | {avg_missing:.2f} | {fc}/{len(rows)} |")

    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_md


def main() -> int:
    started = datetime.now(timezone.utc)
    categories = load_categories()
    print(f"Loaded categories: {len(categories)}")
    app = create_app(base_path=ROOT)

    results: list[CategoryRun] = []
    with TestClient(app) as client:
        for i, (industry, category) in enumerate(categories, start=1):
            print(f"[{i}/{len(categories)}] {industry} / {category} ...", flush=True)
            row = run_one(client, industry, category, exec_idx=i)
            results.append(row)
            status = "OK" if row.success else "ISSUE"
            print(
                f"  -> {status} task={row.task_state} report={row.report_id} "
                f"match={row.match_score} risk={row.risk_score} missing={row.missing_fields} visuals={row.visuals_count}"
            )
            if row.issues:
                print(f"     issues: {row.issues[:3]}")

    out_json, out_md = write_reports(results, started)
    print(f"\nWrote:\n- {out_json}\n- {out_md}")

    issues = [r for r in results if r.issues or not r.success]
    print(f"Summary: {len(results)} total, {len(results)-len(issues)} clean, {len(issues)} with issues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
