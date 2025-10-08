How to add a new LLM provider

1. Create a new file `server/llm/<your_provider>_provider.py`.
2. Add a class that inherits from BaseLLM (or implements `respond(prompt: str) -> str`).
3. Register your provider with LLMFactory by calling:

    from .factory import LLMFactory
    LLMFactory.register_provider("your_name", YourProviderClass)

Example (minimal):

```py
from .factory import BaseLLM, LLMFactory

class DemoLLM(BaseLLM):
    def respond(self, prompt: str) -> str:
        return f"demo response to: {prompt}"

LLMFactory.register_provider("demo", DemoLLM)
```

Use `provider` field in `/llm/respond` to select a provider.
