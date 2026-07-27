from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.backend.app import _toji_unavailable_detail
from src.backend.llm.orchestrator import LLMOrchestrator
from src.backend.llm.providers import LLMResult, OllamaProvider


class _FakeResp:
    def __init__(self, status_code: int, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class _DummyProvider:
    name = "ollama"
    model = "dummy"

    def is_available(self) -> bool:
        return True

    def chat(self, *args, **kwargs):
        return None


class OllamaIntakeResilienceTests(unittest.TestCase):
    def test_ollama_provider_keeps_cloud_model_without_alias_fallback(self):
        env = {
            "OLLAMA_BASE_URL": "https://ollama.com",
            "OLLAMA_MODEL": "gpt-oss:120b-cloud",
            "OLLAMA_API_KEY": "x",
            "TOJI_OLLAMA_CLOUD_ONLY": "1",
            "OLLAMA_DISABLE_THINKING": "1",
            "OLLAMA_EMPTY_CONTENT_RETRY": "0",
            "OLLAMA_CLOUD_ALIAS_FALLBACK": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            provider = OllamaProvider()

        calls: list[str] = []

        def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
            calls.append(str((json or {}).get("model") or ""))
            return _FakeResp(404, {"error": "model not found"}, "model not found")

        provider._http.post = _fake_post  # type: ignore[method-assign]
        result = provider.chat("sys", "user")
        self.assertIsNone(result)
        self.assertEqual(calls, ["gpt-oss:120b-cloud"])
        self.assertTrue(str(provider.last_error or "").strip())

    def test_ollama_provider_retries_final_when_reasoning_only(self):
        env = {
            "OLLAMA_BASE_URL": "https://ollama.com",
            "OLLAMA_MODEL": "gpt-oss:120b-cloud",
            "OLLAMA_API_KEY": "x",
            "TOJI_OLLAMA_CLOUD_ONLY": "1",
            "OLLAMA_DISABLE_THINKING": "1",
            "OLLAMA_EMPTY_CONTENT_RETRY": "0",
            "OLLAMA_CLOUD_ALIAS_FALLBACK": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            provider = OllamaProvider()

        calls: list[dict] = []

        def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001
            payload = dict(json or {})
            calls.append(payload)
            if len(calls) == 1:
                return _FakeResp(200, {"message": {"content": "", "thinking": "analysis..."}})
            return _FakeResp(200, {"message": {"content": "Final answer."}})

        provider._http.post = _fake_post  # type: ignore[method-assign]
        result = provider.chat("sys", "user")
        self.assertIsNotNone(result)
        self.assertEqual(result.model, "gpt-oss:120b-cloud")
        self.assertEqual(result.content, "Final answer.")
        self.assertEqual(len(calls), 2)
        self.assertFalse(provider.last_error)

    def test_unavailable_detail_surfaces_model_not_found_hint(self):
        provider = SimpleNamespace(transport_target="cloud", api_key="x", last_error="Ollama chat HTTP 404 for model")
        orchestrator_stub = SimpleNamespace(provider_name="ollama", provider=provider)
        detail = _toji_unavailable_detail(orchestrator_stub, fallback="fallback")
        self.assertIn("could not find the configured model", detail.lower())

    def test_pure_intake_turn_rewrite_failure_is_non_fatal(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        base_payload = (
            '{"assistant_message":"Noted.","ready_to_analyze":false,'
            '"parse_confidence":0.6,"captured_facts":[],"time_context":""}'
        )

        sequence = [
            LLMResult(content=base_payload, provider="ollama", model="gpt-oss:120b-cloud"),
            None,  # first-person rewrite intentionally fails
        ]

        def _fake_chat_with_persona(**kwargs):  # noqa: ARG001
            return sequence.pop(0) if sequence else None

        with patch.object(llm, "llm_available", return_value=True), patch.object(
            llm, "_chat_with_persona", side_effect=_fake_chat_with_persona
        ), patch.dict(os.environ, {"TOJI_INTAKE_FAST_MODE": "0"}, clear=False):
            out = llm.pure_intake_chat_turn(
                industry="ecommerce",
                category="general_operations",
                payload_context={"question_limit": 5, "questions_answered": 0},
                transcript=[],
                user_message="",
            )

        self.assertIsInstance(out, dict)
        self.assertTrue(str(out.get("assistant_message") or "").strip())
        self.assertIn("ready_to_analyze", out)

    def test_synthesis_plan_recovers_from_raw_python_script_output(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        raw_script = (
            "def generate_dataframe(n_rows, columns, context):\n"
            "    import pandas as pd\n"
            "    return pd.DataFrame([{c: 1 for c in columns}] * int(n_rows))\n"
        )

        with patch.object(llm, "llm_available", return_value=True), patch.object(
            llm,
            "_chat_with_persona",
            return_value=LLMResult(content=raw_script, provider="ollama", model="gpt-oss:120b-cloud"),
        ):
            plan = llm._llm_synthesis_python_plan(
                industry="bpo",
                category="general_operations",
                columns=["a", "b"],
                user_context="high churn and pricing pressure",
                n_rows=10,
            )

        self.assertIsNotNone(plan)
        self.assertIn("script", plan or {})
        self.assertIn("generate_dataframe", str((plan or {}).get("script") or ""))

    def test_synthesis_plan_direct_script_recovery_when_json_repair_fails(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        direct_script = (
            "def generate_dataframe(n_rows, columns, context):\n"
            "    import pandas as pd\n"
            "    return pd.DataFrame([{c: 1 for c in columns}] * int(n_rows))\n"
        )
        calls = [
            LLMResult(content="not json and no code", provider="ollama", model="gpt-oss:120b-cloud"),
            None,  # json repair attempt fails
            LLMResult(content=direct_script, provider="ollama", model="gpt-oss:120b-cloud"),
        ]

        def _fake_chat_with_persona(**kwargs):  # noqa: ARG001
            return calls.pop(0) if calls else None

        with patch.object(llm, "llm_available", return_value=True), patch.object(
            llm, "_chat_with_persona", side_effect=_fake_chat_with_persona
        ):
            plan = llm._llm_synthesis_python_plan(
                industry="bpo",
                category="general_operations",
                columns=["a", "b"],
                user_context="high churn and pricing pressure",
                n_rows=10,
            )

        self.assertIsNotNone(plan)
        self.assertIn("generate_dataframe", str((plan or {}).get("script") or ""))
        trace = [str(x) for x in ((plan or {}).get("analysis_trace") or [])]
        self.assertTrue(any("direct_script_recovery_applied" in row for row in trace))

    def test_upload_transform_plan_direct_script_recovery_when_json_repair_fails(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        direct_script = (
            "def transform_dataframe(df, context):\n"
            "    return df.copy()\n"
        )
        calls = [
            LLMResult(content="not json and no code", provider="ollama", model="gpt-oss:120b-cloud"),
            None,  # json repair attempt fails
            LLMResult(content=direct_script, provider="ollama", model="gpt-oss:120b-cloud"),
        ]

        def _fake_chat_with_persona(**kwargs):  # noqa: ARG001
            return calls.pop(0) if calls else None

        with patch.object(llm, "llm_available", return_value=True), patch.object(
            llm, "_chat_with_persona", side_effect=_fake_chat_with_persona
        ):
            plan = llm._llm_upload_transform_python_plan(
                industry="bpo",
                category="general_operations",
                user_context="uploaded csv with churn and seat data",
                dataset_profile={"rows": 100, "columns": [{"name": "x"}]},
                context={"foo": "bar"},
            )

        self.assertIsNotNone(plan)
        self.assertIn("transform_dataframe", str((plan or {}).get("script") or ""))
        trace = [str(x) for x in ((plan or {}).get("analysis_trace") or [])]
        self.assertTrue(any("direct_script_recovery_applied" in row for row in trace))

    def test_execute_synthesis_script_accepts_tuple_and_partial_columns(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        columns = ["event_date", "demand_volume", "orders_count"]
        script = (
            "def generate_dataframe(n_rows, columns, context):\n"
            "    import pandas as pd\n"
            "    df = pd.DataFrame({\n"
            "        'eventdate': ['2026-01-01'] * int(n_rows),\n"
            "        'demandvolume': [120.0] * int(n_rows)\n"
            "    })\n"
            "    return (df, {'meta': 'ok'})\n"
        )

        out = llm._execute_synthesis_script(
            script=script,
            n_rows=25,
            columns=columns,
            context_payload={"industry": "bpo"},
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(len(out), 25)
        self.assertListEqual(list(out.columns), columns)
        self.assertTrue((out["demand_volume"] == 120.0).all())
        self.assertTrue(out["orders_count"].isna().all())

    def test_execute_upload_transform_script_falls_back_to_input_when_empty(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        script = (
            "def transform_dataframe(df, context):\n"
            "    import pandas as pd\n"
            "    return pd.DataFrame()\n"
        )
        inp = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = llm._execute_upload_transform_script(
            script=script,
            df=inp,
            context_payload={},
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(len(out), len(inp))
        self.assertListEqual(list(out.columns), list(inp.columns))

    def test_execute_synthesis_script_accepts_alt_callable_and_common_imports(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        columns = ["event_date", "seat_churn"]
        script = (
            "import json\n"
            "import re\n"
            "from collections import defaultdict\n"
            "def create_dataframe(n_rows, columns, context):\n"
            "    _ = json.dumps({'rows': int(n_rows)})\n"
            "    _ = re.sub(r'[^a-z]+', '', 'seat_churn')\n"
            "    bag = defaultdict(int)\n"
            "    bag['x'] += 1\n"
            "    return pd.DataFrame({columns[0]: ['2026-01-01'] * int(n_rows), columns[1]: [130] * int(n_rows)})\n"
            "if __name__ == '__main__':\n"
            "    raise RuntimeError('should not execute in sandbox runtime')\n"
        )

        out = llm._execute_synthesis_script(
            script=script,
            n_rows=20,
            columns=columns,
            context_payload={"industry": "bpo"},
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(len(out), 20)
        self.assertListEqual(list(out.columns), columns)
        self.assertTrue((out["seat_churn"] == 130).all())

    def test_execute_upload_transform_script_accepts_alt_callable_name(self):
        root = Path(__file__).resolve().parents[2]
        with patch("src.backend.llm.orchestrator.build_provider", return_value=_DummyProvider()):
            llm = LLMOrchestrator(root)

        script = (
            "import re\n"
            "def normalize_dataframe(df, context):\n"
            "    out = df.copy()\n"
            "    out.columns = [re.sub(r'\\s+', '_', str(c).strip().lower()) for c in out.columns]\n"
            "    return out\n"
        )
        inp = pd.DataFrame({"Seat Churn": [130, 129], "New Seats": [210, 215]})
        out = llm._execute_upload_transform_script(
            script=script,
            df=inp,
            context_payload={"industry": "bpo"},
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertListEqual(list(out.columns), ["seat_churn", "new_seats"])
        self.assertEqual(len(out), len(inp))


if __name__ == "__main__":
    unittest.main()
