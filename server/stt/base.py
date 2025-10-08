"""
Base class for Speech-to-Text providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class STTProvider(ABC):
    """Abstract base class for STT providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT provider.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or {}
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio (default: 16000)
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def transcribe_streaming(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a streaming audio chunk.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            sample_rate: Sample rate of the audio (default: 16000)
            
        Returns:
            Transcribed text for this chunk
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str) -> None:
        """
        Set the language for transcription.
        
        Args:
            language: Language code (e.g., 'en', 'hi', 'gu')
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(config)
