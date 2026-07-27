#!/usr/bin/env python3
"""Evaluate Toji fine-tune behavior on structured task metrics.

Supported runtimes:
- reference: returns ground-truth assistant (harness sanity check only)
- ollama: local Ollama model inference
- huggingface: HF model or base+adapter inference

Metrics:
- parse_accuracy (intake_parse)
- mapping_top1 / mapping_top3 (alias + column mapping tasks)
- worker_selection_accuracy (worker/routing tasks)
- grounded_response_score (chat grounded/guardrail tasks)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests


EXPECTED_JSON_TASKS = {
    "intake_parse",
    "routing_decision",
    "column_mapping_decision",
    "alias_mapping",
    "target_semantic_guess",
    "target_semantics_resolution",
    "worker_selection",
    "l1_synthesis",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s\.\-:%/]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _word_f1(pred: str, truth: str) -> float:
    p = _normalize_text(pred).split()
    t = _normalize_text(truth).split()
    if not p or not t:
        return 0.0
    pc = Counter(p)
    tc = Counter(t)
    overlap = sum((pc & tc).values())
    if overlap <= 0:
        return 0.0
    prec = overlap / max(1, len(p))
    rec = overlap / max(1, len(t))
    return (2.0 * prec * rec) / max(1e-9, prec + rec)


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value or "").strip().replace(",", "")
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None


def _maybe_parse_json(text: str) -> Optional[Any]:
    txt = str(text or "").strip()
    if not txt:
        return None

    # Direct parse.
    try:
        return json.loads(txt)
    except Exception:
        pass

    # Strip code fences.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE | re.DOTALL).strip()
    if fenced != txt:
        try:
            return json.loads(fenced)
        except Exception:
            pass

    # Extract first object/list.
    obj_match = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass
    arr_match = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass
    return None


def _deterministic_sample(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return rows
    scored: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        token = _hash_text(f"{seed}:{json.dumps(row, sort_keys=True, ensure_ascii=True)}")
        scored.append((token, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:cap]]


def _norm_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"assistant", "user", "system"}:
        return role
    return "user"


def _render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = _norm_role(msg.get("role"))
        content = str(msg.get("content") or "").strip()
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts).strip() + "\n"


@dataclass
class EvalSample:
    sample_id: str
    task_type: str
    prompt: str
    expected: str
    metadata: dict[str, Any]
    source_row: dict[str, Any]


def _load_samples(path: Path, cap: int, seed: int) -> list[EvalSample]:
    rows = list(_iter_jsonl(path))
    rows = _deterministic_sample(rows, cap, seed=seed)
    out: list[EvalSample] = []
    for row in rows:
        sample_id = str(row.get("id") or "")[:64]
        task_type = str(row.get("task_type") or "unknown")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue

        last_assistant_idx = -1
        for idx, msg in enumerate(msgs):
            if isinstance(msg, dict) and _norm_role(msg.get("role")) == "assistant":
                last_assistant_idx = idx
        if last_assistant_idx < 0:
            continue

        expected = str((msgs[last_assistant_idx] or {}).get("content") or "").strip()
        if not expected:
            continue

        prompt_msgs = [m for m in msgs[:last_assistant_idx] if isinstance(m, dict)]
        prompt_msgs.append({"role": "assistant", "content": ""})
        prompt = _render_messages(prompt_msgs)

        if not sample_id:
            sample_id = _hash_text(prompt + "\n" + expected)[:24]
        out.append(
            EvalSample(
                sample_id=sample_id,
                task_type=task_type,
                prompt=prompt,
                expected=expected,
                metadata=metadata,
                source_row=row,
            )
        )
    return out


def _mapping_candidates(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        direct = payload.get("canonical_column")
        if isinstance(direct, str) and direct.strip():
            out.append(direct.strip())
        for key in ("top_candidates", "candidate_columns", "candidates"):
            vals = payload.get(key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                    elif isinstance(v, dict):
                        for k in ("canonical_column", "column", "name"):
                            vv = v.get(k)
                            if isinstance(vv, str) and vv.strip():
                                out.append(vv.strip())
        mappings = payload.get("column_mappings")
        if isinstance(mappings, list):
            for row in mappings:
                if isinstance(row, dict):
                    for k in ("canonical_column", "to", "mapped_to", "canonical"):
                        vv = row.get(k)
                        if isinstance(vv, str) and vv.strip():
                            out.append(vv.strip())
                    cands = row.get("candidates")
                    if isinstance(cands, list):
                        for c in cands:
                            if isinstance(c, str) and c.strip():
                                out.append(c.strip())
                            elif isinstance(c, dict):
                                cc = c.get("canonical_column")
                                if isinstance(cc, str) and cc.strip():
                                    out.append(cc.strip())
    return out


def _extract_worker_ids(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        direct = payload.get("worker_model_id")
        if isinstance(direct, str) and direct.strip():
            out.append(direct.strip())
        for key in ("top_workers", "worker_hypotheses"):
            vals = payload.get(key)
            if isinstance(vals, list):
                for row in vals:
                    if isinstance(row, dict):
                        vv = row.get("worker_model_id")
                        if isinstance(vv, str) and vv.strip():
                            out.append(vv.strip())
    return out


class ReferenceRuntime:
    def generate(self, sample: EvalSample, _: float, __: int) -> str:
        return sample.expected


class OllamaRuntime:
    def __init__(self, base_url: str, model: str, timeout_sec: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def generate(self, sample: EvalSample, temperature: float, max_new_tokens: int) -> str:
        payload = {
            "model": self.model,
            "prompt": sample.prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        }
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        row = r.json()
        return str(row.get("response") or "").strip()


class HuggingFaceRuntime:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: Optional[str],
        trust_remote_code: bool,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers + torch are required for huggingface runtime.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dtype = torch.float16
            device_map = None
        else:
            dtype = torch.float32
            device_map = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError("peft is required when --adapter-path is provided.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)

        if device_map is None:
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                model = model.to("mps")
            else:
                model = model.to("cpu")
        model.eval()
        self.model = model

    def generate(self, sample: EvalSample, temperature: float, max_new_tokens: int) -> str:
        tok = self.tokenizer(sample.prompt, return_tensors="pt")
        dev = next(self.model.parameters()).device
        tok = {k: v.to(dev) for k, v in tok.items()}
        with self.torch.no_grad():
            out = self.model.generate(
                **tok,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][tok["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _score_sample(task_type: str, expected: str, predicted: str) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {
        "exact_match": None,
        "json_valid": None,
        "parse_accuracy": None,
        "mapping_top1": None,
        "mapping_top3": None,
        "worker_selection_accuracy": None,
        "grounded_response_score": None,
    }
    out["exact_match"] = 1.0 if _normalize_text(expected) == _normalize_text(predicted) else 0.0

    expected_json = _maybe_parse_json(expected)
    predicted_json = _maybe_parse_json(predicted)

    if task_type in EXPECTED_JSON_TASKS:
        out["json_valid"] = 1.0 if predicted_json is not None else 0.0

    if task_type == "intake_parse":
        if isinstance(expected_json, dict) and isinstance(predicted_json, dict):
            comps: list[float] = []
            exp_source = str(expected_json.get("answer_source") or "").strip().lower()
            pred_source = str(predicted_json.get("answer_source") or "").strip().lower()
            if exp_source:
                comps.append(1.0 if exp_source == pred_source else 0.0)

            exp_norm = _to_float(expected_json.get("normalized_answer"))
            pred_norm = _to_float(predicted_json.get("normalized_answer"))
            if exp_norm is not None and pred_norm is not None:
                denom = max(1.0, abs(exp_norm))
                rel_err = abs(pred_norm - exp_norm) / denom
                comps.append(max(0.0, 1.0 - min(1.0, rel_err)))

            exp_pc = _to_float(expected_json.get("parse_confidence"))
            pred_pc = _to_float(predicted_json.get("parse_confidence"))
            if exp_pc is not None and pred_pc is not None:
                comps.append(max(0.0, 1.0 - min(1.0, abs(pred_pc - exp_pc))))

            out["parse_accuracy"] = (sum(comps) / len(comps)) if comps else 0.0
        else:
            out["parse_accuracy"] = _word_f1(predicted, expected)

    if task_type in {"alias_mapping", "column_mapping_decision"}:
        exp_candidates = _mapping_candidates(expected_json)
        pred_candidates = _mapping_candidates(predicted_json)
        if exp_candidates and pred_candidates:
            target = exp_candidates[0]
            out["mapping_top1"] = 1.0 if target == pred_candidates[0] else 0.0
            out["mapping_top3"] = 1.0 if target in pred_candidates[:3] else 0.0
        else:
            out["mapping_top1"] = _word_f1(predicted, expected)
            out["mapping_top3"] = out["mapping_top1"]

    if task_type in {"worker_selection", "routing_decision"}:
        exp_workers = _extract_worker_ids(expected_json)
        pred_workers = _extract_worker_ids(predicted_json)
        if exp_workers and pred_workers:
            target = exp_workers[0]
            out["worker_selection_accuracy"] = 1.0 if target in pred_workers[:3] else 0.0
        else:
            out["worker_selection_accuracy"] = _word_f1(predicted, expected)

    if task_type in {"chat_grounded_response", "chat_guardrail"}:
        out["grounded_response_score"] = _word_f1(predicted, expected)

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-file", default="data/toji_finetune/toji_v2_messages_test.jsonl")
    p.add_argument("--max-samples", type=int, default=200, help="0 means all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runtime", default="ollama", help="ollama|huggingface|reference")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--out", default="outputs/toji_eval/latest_eval.json")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--predictions-out", default="outputs/toji_eval/latest_predictions.jsonl")
    p.add_argument("--ollama-base-url", default="http://localhost:11434")
    p.add_argument("--ollama-model", default="qwen2.5:7b")
    p.add_argument("--ollama-timeout-sec", type=int, default=120)
    p.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-path", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    test_path = Path(args.test_file).resolve()
    if not test_path.exists():
        raise SystemExit(f"Test file not found: {test_path}")

    samples = _load_samples(test_path, cap=args.max_samples, seed=args.seed)
    if not samples:
        raise SystemExit("No evaluable samples found in test file.")

    runtime_name = str(args.runtime or "").strip().lower()
    runtime_note = None
    if runtime_name == "reference":
        runtime = ReferenceRuntime()
        runtime_note = (
            "reference mode echoes expected answers and is for harness checks only; "
            "do not use it to estimate real model quality."
        )
    elif runtime_name == "ollama":
        runtime = OllamaRuntime(
            base_url=args.ollama_base_url,
            model=args.ollama_model,
            timeout_sec=args.ollama_timeout_sec,
        )
    elif runtime_name == "huggingface":
        runtime = HuggingFaceRuntime(
            model_name_or_path=args.hf_model,
            adapter_path=(args.adapter_path or None),
            trust_remote_code=bool(args.trust_remote_code),
        )
    else:
        raise SystemExit(f"Unsupported runtime: {args.runtime}")

    agg: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    predictions: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []

    for sample in samples:
        t0 = time.time()
        predicted = runtime.generate(sample, args.temperature, args.max_new_tokens)
        latency_ms = (time.time() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        task_counts[sample.task_type] += 1
        scores = _score_sample(sample.task_type, sample.expected, predicted)
        for k, v in scores.items():
            if v is None:
                continue
            agg[k].append(v)
            per_task[sample.task_type][k].append(v)

        if args.save_predictions:
            predictions.append(
                {
                    "id": sample.sample_id,
                    "task_type": sample.task_type,
                    "predicted": predicted,
                    "expected": sample.expected,
                    "scores": scores,
                    "latency_ms": round(latency_ms, 2),
                }
            )

    summary_metrics = {
        key: round(sum(vals) / len(vals), 4) for key, vals in agg.items() if vals
    }
    per_task_summary: dict[str, dict[str, float]] = {}
    for task, metrics in per_task.items():
        per_task_summary[task] = {
            key: round(sum(vals) / len(vals), 4) for key, vals in metrics.items() if vals
        }

    out = {
        "generated_at": _now_iso(),
        "runtime": runtime_name,
        "runtime_note": runtime_note,
        "samples_evaluated": len(samples),
        "task_counts": dict(task_counts),
        "latency_ms_avg": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "summary_metrics": summary_metrics,
        "per_task_metrics": per_task_summary,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    if args.save_predictions:
        pred_path = Path(args.predictions_out).resolve()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        out["predictions_out"] = str(pred_path)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
