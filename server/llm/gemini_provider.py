import os
import json
from typing import Optional, List, Dict
from .factory import BaseLLM

# Prefer the official GenAI SDK if available
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


class GeminiLLM(BaseLLM):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        # model can be provided via config or env var
        self.model = (config.get("model") if config else None) or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self.api_key = os.getenv("GEMINI_API_KEY")

        # Use genai client if available
        self.client = None
        if GENAI_AVAILABLE:
            try:
                # genai.Client will pick up env var if not provided
                self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
            except Exception:
                self.client = None

    def respond(self, prompt: str, history: Optional[List[Dict]] = None) -> str:
        # Prefer SDK client
        if not GENAI_AVAILABLE:
            return "[genai SDK not installed]"
        if not self.client:
            return "[genai client not initialized]"

        # Build contents in the SDK-friendly structure
        contents = []
        if history:
            for turn in history:
                contents.append({
                    "role": turn.get("role", "user"),
                    "parts": [{"text": turn.get("text", "")}],
                })

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        try:
            # Use generation config if provided in config
            gen_config = None
            if self.config and "generation_kwargs" in self.config:
                gen_config = self.config["generation_kwargs"]

            # The genai SDK supports passing contents as a string or list; we use list for structure
            resp = self.client.models.generate_content(model=self.model, contents=contents, config=gen_config)

            # The SDK commonly exposes a `.text` attribute for simple use
            if hasattr(resp, "text") and resp.text:
                return resp.text

            # Fallbacks for various shapes
            try:
                # Some SDK responses include output or candidates
                if hasattr(resp, "output"):
                    out = resp.output
                    if isinstance(out, list) and out:
                        # try to pick first text-looking field
                        first = out[0]
                        if isinstance(first, dict):
                            parts = first.get("content") or first.get("parts")
                            if isinstance(parts, list) and parts:
                                p0 = parts[0]
                                if isinstance(p0, dict) and "text" in p0:
                                    return p0["text"]
                # last resort stringify
                return json.dumps(resp)
            except Exception:
                return str(resp)

        except Exception as e:
            return f"[gemini genai request failed] {e}"


# Register provider automatically when module is imported
try:
    from .factory import LLMFactory
    LLMFactory.register_provider("gemini", GeminiLLM)
except Exception:
    pass
