"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        thinking_disabled = False

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                thinking_disabled = True
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field, so both
                # are needed to actually stop a thinking-capable model from
                # reasoning (#25758). Endpoints that recognize neither simply
                # ignore them.
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
                top_level["reasoning_effort"] = _effort

        # Vertex OpenAPI behind provider=custom never goes through VertexProfile,
        # so Gemini hides thought parts unless we emit thinking_config here.
        model = ctx.get("model") or ""
        base_url = str(ctx.get("base_url") or self.base_url or "")
        if (
            not thinking_disabled
            and "aiplatform.googleapis.com" in base_url
            and "gemini" in str(model).lower()
        ):
            from agent.transports.chat_completions import (
                _build_gemini_thinking_config,
                _snake_case_gemini_thinking_config,
            )

            rc = reasoning_config if isinstance(reasoning_config, dict) else {
                "enabled": True,
                "effort": "medium",
            }
            raw = _build_gemini_thinking_config(model, rc)
            thinking = _snake_case_gemini_thinking_config(raw)
            if thinking:
                extra_body["google"] = {"thinking_config": thinking}

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
