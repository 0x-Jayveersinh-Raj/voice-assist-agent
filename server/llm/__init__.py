"""LLM package init

Import providers here to ensure they register with the factory at package import time.
"""

from .factory import LLMFactory

# Import known providers so they auto-register on package import
try:
	from . import gemini_provider  # noqa: F401
except Exception:
	# Import failures shouldn't break package import
	pass

__all__ = ["LLMFactory"]
