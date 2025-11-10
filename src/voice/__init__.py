"""
Voice services for interview system.

Includes:
- Text-to-Speech (TTS): Read questions with AI voices
- Speech-to-Text (STT): Voice input for answers
- Emotion Detection: Analyze voice emotions (future feature)
"""

from .tts_service import TTSService, get_tts_service
from .stt_service import STTService, get_stt_service

__all__ = [
    'TTSService',
    'get_tts_service',
    'STTService',
    'get_stt_service'
]

