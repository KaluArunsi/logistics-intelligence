"""
Provider adapter for Ollama LLM runtime — the only supported provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Generator, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def _is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _is_loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return _is_loopback_host(parsed.hostname or "")
    except Exception:
        return False


def _extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1].strip()
                return candidate
    # Last chance: strip markdown fence and retry.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    return ""


@dataclass
class LLMResult:
    content: str
    provider: str
    model: str


class BaseProvider:
    name = "none"
    model = "none"

    def is_enabled(self) -> bool:
        return False

    def chat(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> Optional[LLMResult]:
        return None

    def chat_stream(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> Generator[str, None, None]:
        """Streaming fallback: calls chat() and yields the full content in one chunk."""
        result = self.chat(system_prompt, user_prompt, json_mode=json_mode, temperature=temperature)
        if result and result.content:
            yield result.content

    def is_available(self) -> bool:
        return self.is_enabled()


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self):
        self.api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        # In cloud deploys (Railway), default to cloud transport unless explicitly overridden.
        is_railway = any(
            os.getenv(k, "").strip()
            for k in ("RAILWAY_PROJECT_ID", "RAILWAY_SERVICE_ID", "RAILWAY_ENVIRONMENT_NAME", "RAILWAY_STATIC_URL")
        )
        cloud_only_raw = os.getenv("TOJI_OLLAMA_CLOUD_ONLY", "").strip().lower()
        if cloud_only_raw:
            self.cloud_only = cloud_only_raw in {"1", "true", "yes", "on"}
        else:
            self.cloud_only = bool(is_railway)
        self.allow_local_with_key = os.getenv("TOJI_OLLAMA_ALLOW_LOCAL_WITH_KEY", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        base_from_env = os.getenv("OLLAMA_BASE_URL", "").strip()
        # Backward-compatible alias: many setups still use OLLAMA_HOST.
        host_alias = os.getenv("OLLAMA_HOST", "").strip()
        default_base = "http://localhost:11434"
        if self.api_key:
            # With API key mode enabled, never silently route to localhost unless explicitly allowed.
            if base_from_env and _is_loopback_url(base_from_env) and not self.allow_local_with_key:
                selected_base = "https://ollama.com"
            else:
                selected_base = base_from_env or "https://ollama.com"
        elif self.cloud_only:
            selected_base = base_from_env or "https://ollama.com"
        else:
            selected_base = base_from_env or host_alias or default_base
        if "://" not in selected_base:
            selected_base = f"http://{selected_base}"
        selected_base = selected_base.rstrip("/")
        if selected_base.endswith("/api"):
            # Support users who configure OLLAMA_BASE_URL=https://ollama.com/api.
            selected_base = selected_base[: -len("/api")]
        self.base_url = selected_base.rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT_SEC", "90"))
        self.chat_num_predict = max(0, int(os.getenv("OLLAMA_CHAT_NUM_PREDICT", "4096")))
        self.gpt_oss_think_level = os.getenv("OLLAMA_GPT_OSS_THINK_LEVEL", "medium").strip().lower() or "medium"
        if self.gpt_oss_think_level not in {"low", "medium", "high"}:
            self.gpt_oss_think_level = "medium"
        self.disable_thinking = os.getenv("OLLAMA_DISABLE_THINKING", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Keep cloud model selection strict by default: do not silently degrade
        # `*-cloud` model tags to local/base aliases unless explicitly enabled.
        self.allow_cloud_alias_fallback = os.getenv("OLLAMA_CLOUD_ALIAS_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.empty_content_retry = os.getenv("OLLAMA_EMPTY_CONTENT_RETRY", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.auth_header = os.getenv("OLLAMA_AUTH_HEADER", "Authorization").strip() or "Authorization"
        self._http = requests.Session()
        self.last_error: str = ""
        parsed_host = (urlparse(self.base_url).hostname or "").strip().lower()
        self.transport_target = "local" if _is_loopback_host(parsed_host) else "cloud"

    @staticmethod
    def _is_gpt_oss_model(model_name: str) -> bool:
        return "gpt-oss" in str(model_name or "").strip().lower()
        logger.info(
            "Ollama provider configured: target=%s base_url=%s model=%s cloud_only=%s api_key_present=%s disable_thinking=%s chat_num_predict=%d empty_content_retry=%s",
            self.transport_target,
            self.base_url,
            self.model,
            str(self.cloud_only).lower(),
            "yes" if bool(self.api_key) else "no",
            "yes" if self.disable_thinking else "no",
            self.chat_num_predict,
            "yes" if self.empty_content_retry else "no",
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.auth_header] = f"Bearer {self.api_key}"
        return headers

    def _set_error(self, message: str) -> None:
        self.last_error = str(message or "").strip()[:800]

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        self.last_error = ""
        if self.transport_target == "cloud" and not self.api_key:
            logger.warning("Ollama cloud transport selected but OLLAMA_API_KEY is missing.")
            self._set_error("Ollama cloud API key is missing.")
            return False
        try:
            logger.info("Ollama availability probe: base_url=%s model=%s", self.base_url, self.model)
            resp = self._http.get(f"{self.base_url}/api/tags", headers=self._headers(), timeout=3)
            logger.info("Ollama availability probe result: status=%s", resp.status_code)
            if resp.status_code in (401, 403):
                logger.warning("Ollama availability probe unauthorized: HTTP %s", resp.status_code)
                self._set_error("Service authentication failed.")
                return False
            # Cloud gateways can return non-2xx for /api/tags while /api/chat still works.
            # Treat non-5xx as reachable runtime to avoid false "unavailable" status.
            if resp.status_code >= 500:
                logger.warning("Ollama availability probe server error: HTTP %s", resp.status_code)
                self._set_error("Service temporarily unavailable.")
            return resp.status_code < 500
        except Exception as exc:
            logger.warning("Ollama availability probe failed: %s", exc)
            self._set_error("Service temporarily unavailable.")
            return False

    def chat(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> Optional[LLMResult]:
        self.last_error = ""
        options: dict[str, object] = {"temperature": temperature}
        if self.chat_num_predict > 0:
            options["num_predict"] = self.chat_num_predict
        configured_model = str(self.model).strip()
        model_candidates = [configured_model]
        if configured_model.endswith("-cloud") and self.allow_cloud_alias_fallback:
            base_candidate = configured_model[: -len("-cloud")]
            if base_candidate and base_candidate not in model_candidates:
                model_candidates.append(base_candidate)

        logger.info(
            "Ollama chat request: base_url=%s model=%s json_mode=%s system_chars=%d user_chars=%d",
            self.base_url,
            configured_model,
            json_mode,
            len(system_prompt or ""),
            len(user_prompt or ""),
        )

        for model_name in model_candidates:
            payload = {
                "model": model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": options,
            }
            if self._is_gpt_oss_model(model_name):
                payload["think"] = self.gpt_oss_think_level
            elif self.disable_thinking:
                payload["think"] = False
            if json_mode:
                payload["format"] = "json"

            try:
                resp = self._http.post(
                    f"{self.base_url}/api/chat",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except Exception as exc:
                logger.warning("Ollama chat request failed: model=%s err=%s", model_name, exc)
                self._set_error("LLM provider unavailable.")
                continue

            logger.info("Ollama chat response status=%s model=%s", resp.status_code, model_name)
            if resp.status_code >= 300:
                body_preview = str(resp.text or "")[:300]
                logger.warning("Ollama chat non-success status=%s model=%s body=%s", resp.status_code, model_name, body_preview)
                self._set_error("LLM provider unavailable.")
                continue

            try:
                data = resp.json()
            except Exception as exc:
                logger.warning("Ollama chat invalid JSON response: model=%s err=%s", model_name, exc)
                self._set_error("LLM provider returned an invalid response.")
                continue

            message = data.get("message") or {}
            content = str(message.get("content") or "").strip()
            if not content:
                logger.warning("Ollama chat returned empty content for model=%s", model_name)
                if self.empty_content_retry:
                    retry_payload = dict(payload)
                    retry_payload.pop("think", None)
                    try:
                        retry = self._http.post(
                            f"{self.base_url}/api/chat",
                            headers=self._headers(),
                            json=retry_payload,
                            timeout=self.timeout,
                        )
                        if retry.status_code < 300:
                            retry_data = retry.json()
                            content = str(((retry_data.get("message") or {}).get("content") or "")).strip()
                    except Exception as retry_exc:
                        logger.warning("Ollama chat retry after empty content failed: %s", retry_exc)
                if not content:
                    thinking = str(message.get("thinking") or "").strip()
                    if thinking:
                        # Some cloud responses can contain thinking with empty final content.
                        # Retry once with a stricter final-answer instruction.
                        retry_options = dict(options)
                        retry_options["num_predict"] = max(512, self.chat_num_predict)
                        final_only_payload = dict(payload)
                        final_only_payload["messages"] = [
                            {
                                "role": "system",
                                "content": (
                                    f"{system_prompt}\n\n"
                                    "IMPORTANT: Return only the final answer content. "
                                    "Do not return reasoning/thinking text."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"{user_prompt}\n\n"
                                    "Return only the final answer now. No reasoning."
                                ),
                            },
                        ]
                        if self._is_gpt_oss_model(model_name):
                            final_only_payload["think"] = self.gpt_oss_think_level
                        else:
                            final_only_payload["think"] = False
                        final_only_payload["options"] = retry_options
                        try:
                            final_retry = self._http.post(
                                f"{self.base_url}/api/chat",
                                headers=self._headers(),
                                json=final_only_payload,
                                timeout=self.timeout,
                            )
                            if final_retry.status_code < 300:
                                final_data = final_retry.json()
                                content = str(((final_data.get("message") or {}).get("content") or "")).strip()
                        except Exception as final_retry_exc:
                            logger.warning("Ollama final-only retry failed: %s", final_retry_exc)
                        if not content and json_mode and thinking:
                            extracted_json = _extract_json_object(thinking)
                            if extracted_json:
                                logger.warning(
                                    "Ollama returned thinking-only content; recovered JSON payload from reasoning text for model=%s",
                                    model_name,
                                )
                                content = extracted_json
                        if not content:
                            logger.warning("Ollama returned reasoning-only content for model=%s", model_name)
                            self._set_error("LLM provider returned an empty response.")
                    else:
                        logger.warning("Ollama returned empty content for model=%s", model_name)
                        self._set_error("LLM provider returned an empty response.")
                    if not content:
                        continue

            if json_mode:
                content = content.strip()
                if content.startswith("```"):
                    content = content.strip("`").replace("json\n", "", 1).strip()
                # Keep provider permissive; orchestrator handles JSON repair/recovery.
            self.last_error = ""
            return LLMResult(content=content, provider=self.name, model=model_name)

        return None

    def chat_stream(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> Generator[str, None, None]:
        """Stream response tokens from Ollama using chunked HTTP. Falls back to non-streaming chat() on error."""
        self.last_error = ""
        options: dict[str, object] = {"temperature": temperature}
        if self.chat_num_predict > 0:
            options["num_predict"] = self.chat_num_predict
        model_name = str(self.model).strip()
        payload: dict[str, object] = {
            "model": model_name,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": options,
        }
        if self._is_gpt_oss_model(model_name):
            payload["think"] = self.gpt_oss_think_level
        elif self.disable_thinking:
            payload["think"] = False
        if json_mode:
            payload["format"] = "json"

        resp = None
        try:
            resp = self._http.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            if resp.status_code >= 300:
                resp.close()
                logger.warning("Ollama chat_stream non-success status=%s, falling back to non-streaming", resp.status_code)
                fallback = self.chat(system_prompt, user_prompt, json_mode=json_mode, temperature=temperature)
                if fallback and fallback.content:
                    yield fallback.content
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except Exception:
                    continue
                chunk = str((data.get("message") or {}).get("content") or "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
        except Exception as exc:
            logger.warning("Ollama chat_stream failed: %s — falling back to non-streaming", exc)
            fallback = self.chat(system_prompt, user_prompt, json_mode=json_mode, temperature=temperature)
            if fallback and fallback.content:
                yield fallback.content
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass


def build_provider() -> BaseProvider:
    """Build the Ollama provider — the only supported LLM runtime."""
    return OllamaProvider()
