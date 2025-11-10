"""
Speech-to-Text Service for Voice Input.

Uses OpenAI Whisper for accurate speech recognition.
Supports emotion detection (to be added later).
"""
import os
from typing import Optional, Dict, Any
import tempfile
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("stt_service")

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Install with: pip install openai-whisper")


class STTService:
    """
    Speech-to-Text service for voice input.
    
    Uses OpenAI Whisper for accurate transcription.
    Supports emotion detection (future feature).
    """
    
    def __init__(self):
        """Initialize STT service."""
        self.whisper_available = WHISPER_AVAILABLE
        self.model = None
        
        if self.whisper_available:
            try:
                # Load Whisper model (base model for speed/accuracy balance)
                self.model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                self.whisper_available = False
    
    def speech_to_text(
        self,
        audio_file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert speech audio to text.
        
        Args:
            audio_file_path: Path to audio file
            language: Optional language code (e.g., "en" for English)
        
        Returns:
            Dictionary with:
            - text: Transcribed text
            - language: Detected language
            - confidence: Confidence score (if available)
        """
        if not self.whisper_available or not self.model:
            return {
                "text": "",
                "error": "Whisper not available"
            }
        
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                task="transcribe"
            )
            
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "confidence": 1.0  # Whisper doesn't provide confidence scores directly
            }
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {
                "text": "",
                "error": str(e)
            }
    
    def detect_emotion(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Detect emotion from voice (FUTURE FEATURE).
        
        This is a placeholder for future emotion detection.
        Can use libraries like:
        - pyAudioAnalysis
        - librosa + ML models
        - Azure Speech API (has built-in emotion detection)
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Dictionary with emotion analysis:
            - emotion: detected emotion (happy, sad, neutral, etc.)
            - confidence: confidence score
            - features: extracted audio features
        """
        # TODO: Implement emotion detection
        # For now, return placeholder
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "note": "Emotion detection not yet implemented"
        }


# Singleton instance
_stt_service = None


def get_stt_service() -> STTService:
    """Get or create STT service instance (singleton)."""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service

