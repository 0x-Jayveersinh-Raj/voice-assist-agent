#!/usr/bin/env python3
"""
Example script demonstrating the STT system usage.
"""
import sys
import os
import numpy as np

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from stt.factory import STTFactory


def demonstrate_whisper():
    """Demonstrate Whisper STT provider."""
    print("=== Whisper STT Provider Demo ===")
    
    # Check if Whisper is available
    if 'whisper' not in STTFactory.get_available_providers():
        print("Whisper provider not available (missing dependencies)")
        print("Install with: pip install faster-whisper or pip install openai-whisper")
        return
    
    try:
        # Create Whisper provider
        whisper_provider = STTFactory.create_provider('whisper', {
            'model': 'tiny',
            'language': 'en'
        })
        
        print(f"Provider: {whisper_provider.__class__.__name__}")
        print(f"Supported languages: {whisper_provider.get_supported_languages()[:10]}...")  # Show first 10
        print(f"Current config: {whisper_provider.get_config()}")
        
        # Generate sample audio data (1 second of silence)
        sample_audio = np.zeros(16000, dtype=np.int16)
        
        print("Transcribing sample audio...")
        transcript = whisper_provider.transcribe(sample_audio)
        print(f"Transcript: '{transcript}'")
        
        # Test language switching
        whisper_provider.set_language('hi')
        print(f"Language switched to: {whisper_provider.config.get('language')}")
        
    except Exception as e:
        print(f"Error with Whisper provider: {e}")


def demonstrate_deepgram():
    """Demonstrate Deepgram STT provider."""
    print("\n=== Deepgram STT Provider Demo ===")
    
    # Check if Deepgram is available
    if 'deepgram' not in STTFactory.get_available_providers():
        print("Deepgram provider not available (missing dependencies)")
        print("Install with: pip install deepgram-sdk")
        return
    
    # Check if API key is available
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key:
        print("DEEPGRAM_API_KEY not found in environment variables")
        print("Skipping Deepgram demo")
        return
    
    try:
        # Create Deepgram provider
        deepgram_provider = STTFactory.create_provider('deepgram', {
            'api_key': api_key,
            'language': 'en',
            'model': 'nova-2'
        })
        
        print(f"Provider: {deepgram_provider.__class__.__name__}")
        print(f"Supported languages: {deepgram_provider.get_supported_languages()[:10]}...")  # Show first 10
        print(f"Current config: {deepgram_provider.get_config()}")
        
        # Generate sample audio data (1 second of silence)
        sample_audio = np.zeros(16000, dtype=np.int16)
        
        print("Transcribing sample audio...")
        transcript = deepgram_provider.transcribe(sample_audio)
        print(f"Transcript: '{transcript}'")
        
        # Test language switching
        deepgram_provider.set_language('hi')
        print(f"Language switched to: {deepgram_provider.config.get('language')}")
        
    except Exception as e:
        print(f"Error with Deepgram provider: {e}")


def demonstrate_factory():
    """Demonstrate STT factory functionality."""
    print("\n=== STT Factory Demo ===")
    
    # Get available providers
    providers = STTFactory.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Get provider information
    for provider_type in providers:
        try:
            info = STTFactory.get_provider_info(provider_type)
            print(f"\n{provider_type.upper()} Provider Info:")
            print(f"  Class: {info.get('class', 'Unknown')}")
            if 'supported_languages' in info:
                languages = info['supported_languages']
                print(f"  Languages: {len(languages)} supported")
                print(f"  Sample: {languages[:5]}...")
            if 'error' in info:
                print(f"  Error: {info['error']}")
        except Exception as e:
            print(f"Error getting info for {provider_type}: {e}")


def demonstrate_custom_provider():
    """Demonstrate registering a custom provider."""
    print("\n=== Custom Provider Demo ===")
    
    from stt.base import STTProvider
    
    class MockSTT(STTProvider):
        def __init__(self, config=None):
            super().__init__(config)
            self.transcription_count = 0
        
        def transcribe(self, audio_data, sample_rate=16000):
            self.transcription_count += 1
            return f"Mock transcription #{self.transcription_count}"
        
        def transcribe_streaming(self, audio_chunk, sample_rate=16000):
            return "Mock streaming transcription"
        
        def get_supported_languages(self):
            return ['en', 'mock']
        
        def set_language(self, language):
            self.config['language'] = language
    
    # Register custom provider
    STTFactory.register_provider('mock', MockSTT)
    
    # Create and use custom provider
    mock_provider = STTFactory.create_provider('mock', {'test': 'value'})
    
    print(f"Custom provider: {mock_provider.__class__.__name__}")
    print(f"Config: {mock_provider.get_config()}")
    
    # Test transcription
    sample_audio = np.zeros(1000, dtype=np.int16)
    transcript = mock_provider.transcribe(sample_audio)
    print(f"Transcript: {transcript}")
    
    # Test streaming
    stream_transcript = mock_provider.transcribe_streaming(sample_audio)
    print(f"Streaming transcript: {stream_transcript}")
    
    # Verify it's registered
    providers = STTFactory.get_available_providers()
    print(f"Available providers now include: {providers}")


def main():
    """Main demonstration function."""
    print("STT System Demonstration")
    print("=" * 50)
    
    # Demonstrate factory
    demonstrate_factory()
    
    # Demonstrate Whisper
    demonstrate_whisper()
    
    # Demonstrate Deepgram (if API key available)
    demonstrate_deepgram()
    
    # Demonstrate custom provider
    demonstrate_custom_provider()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")


if __name__ == '__main__':
    main()
