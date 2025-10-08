"""
Factory for creating STT providers.
"""
from typing import Optional, Dict, Any, Type
from .base import STTProvider

# Import providers conditionally
try:
    from .deepgram_provider import DeepgramSTT
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

try:
    from .whisper_provider import WhisperSTT
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class STTFactory:
    """Factory class for creating STT providers."""
    
    _providers = {}
    
    # Register available providers
    if DEEPGRAM_AVAILABLE:
        _providers['deepgram'] = DeepgramSTT
    
    if WHISPER_AVAILABLE:
        _providers['whisper'] = WhisperSTT
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Optional[Dict[str, Any]] = None) -> STTProvider:
        """
        Create an STT provider instance.
        
        Args:
            provider_type: Type of provider ('deepgram', 'whisper')
            config: Configuration dictionary for the provider
            
        Returns:
            STT provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Unsupported provider type: {provider_type}. Available: {available}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available provider types.
        
        Returns:
            List of available provider type names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[STTProvider]) -> None:
        """
        Register a new provider type.
        
        Args:
            name: Name of the provider type
            provider_class: Provider class that inherits from STTProvider
        """
        if not issubclass(provider_class, STTProvider):
            raise ValueError("Provider class must inherit from STTProvider")
        
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider_info(cls, provider_type: str) -> Dict[str, Any]:
        """
        Get information about a provider type.
        
        Args:
            provider_type: Type of provider
            
        Returns:
            Dictionary with provider information
        """
        if provider_type not in cls._providers:
            raise ValueError(f"Provider type '{provider_type}' not found")
        
        provider_class = cls._providers[provider_type]
        
        # Create a temporary instance to get default config and supported languages
        try:
            temp_instance = provider_class()
            return {
                'name': provider_type,
                'class': provider_class.__name__,
                'supported_languages': temp_instance.get_supported_languages(),
                'default_config': temp_instance.get_config()
            }
        except Exception as e:
            return {
                'name': provider_type,
                'class': provider_class.__name__,
                'error': str(e)
            }
