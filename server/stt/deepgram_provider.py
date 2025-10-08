"""
Deepgram STT provider implementation (streaming-focused).
"""
import os
from typing import Optional, Dict, Any
from deepgram import DeepgramClient
from .base import STTProvider


class DeepgramSTT(STTProvider):
    """Deepgram Speech-to-Text provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Deepgram STT provider.
        
        Args:
            config: Configuration dictionary with 'api_key' and optional settings
        """
        super().__init__(config)
        
        # Get API key from config or environment (lazy client init later)
        self.api_key = self.config.get('api_key') or os.getenv('DEEPGRAM_API_KEY')
        self.client = None
        self.language = self.config.get('language', 'en')
        self.model = self.config.get('model', 'nova-2')
        self.smart_format = self.config.get('smart_format', True)
        self.punctuate = self.config.get('punctuate', True)
        
        # Supported languages (subset of Deepgram's supported languages)
        self.supported_languages = [
            'en', 'hi', 'gu', 'bn', 'ta', 'te', 'ml', 'kn', 'mr', 'pa', 'or', 'as',
            'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'tr', 'pl',
            'nl', 'sv', 'da', 'no', 'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl',
            'et', 'lv', 'lt', 'mt', 'cy', 'ga', 'is', 'mk', 'sq', 'sr', 'uk', 'be',
            'ka', 'hy', 'az', 'kk', 'ky', 'uz', 'mn', 'ne', 'si', 'my', 'km', 'lo',
            'th', 'vi', 'id', 'ms', 'tl', 'sw', 'am', 'ha', 'ig', 'yo', 'zu', 'xh',
            'af', 'sq', 'eu', 'ca', 'gl', 'is', 'ga', 'cy', 'mt', 'mk', 'sq', 'sr',
            'uk', 'be', 'ka', 'hy', 'az', 'kk', 'ky', 'uz', 'mn', 'ne', 'si', 'my',
            'km', 'lo', 'th', 'vi', 'id', 'ms', 'tl', 'sw', 'am', 'ha', 'ig', 'yo',
            'zu', 'xh', 'af'
        ]
    
    def transcribe(self, audio_data, sample_rate: int = 16000) -> str:
        """
        Placeholder: Deepgram live streaming is used in the server via WebSocket.
        This method is intentionally a no-op to keep factory/provider listings working.
        """
        return ""
    
    def transcribe_streaming(self, audio_chunk, sample_rate: int = 16000) -> str:
        return ""
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def set_language(self, language: str) -> None:
        """
        Set the language for transcription.
        
        Args:
            language: Language code
        """
        if language in self.supported_languages:
            self.language = language
            self.config['language'] = language
        else:
            raise ValueError(f"Language '{language}' not supported. Supported languages: {self.supported_languages}")
    
    def update_model(self, model: str) -> None:
        """
        Update the Deepgram model.
        
        Args:
            model: Model name (e.g., 'nova-2', 'base', 'enhanced')
        """
        self.model = model
        self.config['model'] = model
