"""Factory for creating LLM providers."""
from typing import Optional, Dict, Any, Type


class BaseLLM:
    """Minimal base class for LLM providers."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def respond(self, prompt: str, history: Optional[list] = None) -> str:
        """Generate a response for the prompt. Implementations may accept an optional conversation history."""
        raise NotImplementedError()


class LLMFactory:
    _providers = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]):
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseLLM:
        if name not in cls._providers:
            available = ', '.join(cls._providers.keys()) or '<none>'
            raise ValueError(f"Unsupported LLM: {name}. Available: {available}")
        return cls._providers[name](config)

    @classmethod
    def get_available_providers(cls):
        return list(cls._providers.keys())
