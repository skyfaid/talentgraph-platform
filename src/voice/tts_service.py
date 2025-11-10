"""
Text-to-Speech Service for Interview Questions.

Uses edge-tts (Microsoft Edge TTS) - FREE, no API key required.
High-quality voices, completely free to use.
"""
import os
import io
import asyncio
from typing import Optional
from pathlib import Path
import tempfile

from ..utils.logger import setup_logger

logger = setup_logger("tts_service")

# Try to import edge-tts (FREE, no API key needed)
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. Install with: pip install edge-tts")

# Try to import gTTS (Google TTS - simple and reliable)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Install with: pip install gtts")


class TTSService:
    """
    Text-to-Speech service for reading interview questions.
    
    Uses edge-tts (Microsoft Edge TTS) - FREE, no API key required.
    High-quality voices, completely free to use.
    """
    
    def __init__(self):
        """Initialize TTS service."""
        self.edge_tts_available = EDGE_TTS_AVAILABLE
        self.gtts_available = GTTS_AVAILABLE
        self.voices_cache = None
        
        if self.edge_tts_available:
            logger.info("TTS Service initialized: Using edge-tts (FREE, no API key needed)")
        elif self.gtts_available:
            logger.info("TTS Service initialized: Using gTTS (Google TTS - simple fallback)")
        else:
            logger.warning("No TTS service available. Install: pip install edge-tts or pip install gtts")
    
    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if not self.edge_tts_available:
            return []
        
        if self.voices_cache is None:
            try:
                # Get edge-tts voices (async function)
                async def _get_voices():
                    voices_list = []
                    voices = await edge_tts.list_voices()
                    for voice in voices:
                        # Filter for English voices (or add language filter)
                        if "en" in voice.get("Locale", "").lower():
                            voices_list.append({
                                "id": voice["ShortName"],
                                "name": voice["ShortName"],
                                "gender": voice.get("Gender", "Unknown"),
                                "locale": voice.get("Locale", "en-US"),
                                "provider": "edge-tts"
                            })
                    return voices_list
                
                # Run async function
                self.voices_cache = asyncio.run(_get_voices())
            except Exception as e:
                logger.error(f"Error getting edge-tts voices: {e}")
                self.voices_cache = []
        
        return self.voices_cache
    
    def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        provider: str = "auto"
    ) -> Optional[bytes]:
        """
        Convert text to speech audio (FREE - no API key needed).
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (optional, uses default if not provided)
            provider: "auto" (tries gTTS first, then edge-tts), "gtts", or "edge-tts"
        
        Returns:
            Audio bytes (MP3 format) or None if failed
        """
        # Auto mode: try edge-tts first (better quality), then gTTS as fallback
        if provider == "auto":
            # Try edge-tts first (better voice quality, more natural)
            if self.edge_tts_available:
                logger.info("Trying edge-tts first (better quality)...")
                result = self._edge_tts(text, voice_id)
                if result:
                    return result
                logger.warning("edge-tts failed, trying gTTS as fallback...")
            
            # Fallback to gTTS (simpler, more reliable)
            if self.gtts_available:
                logger.info("Using gTTS (Google TTS) as fallback...")
                return self._gtts(text)
            
            logger.error("No TTS service available")
            return None
        
        # Specific provider requested
        elif provider == "gtts":
            if not self.gtts_available:
                logger.error("gTTS not available")
                return None
            return self._gtts(text)
        
        elif provider == "edge-tts":
            if not self.edge_tts_available:
                logger.error("edge-tts not available")
                return None
            return self._edge_tts(text, voice_id)
        
        else:
            logger.error(f"Unknown provider: {provider}")
            return None
    
    def _gtts(self, text: str) -> Optional[bytes]:
        """Generate speech using Google TTS (gTTS) - simple and reliable."""
        try:
            # Validate text
            if not text or not text.strip():
                logger.warning("Empty text provided to TTS")
                return None
            
            # Clean text
            text = text.strip()
            
            # Limit text length (gTTS has limits)
            if len(text) > 5000:
                logger.warning(f"Text too long ({len(text)} chars), truncating to 5000")
                text = text[:5000]
            
            logger.info(f"Generating audio with gTTS, text length: {len(text)}")
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to BytesIO buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Get bytes
            audio_data = audio_buffer.getvalue()
            
            if len(audio_data) == 0:
                logger.error("gTTS returned no audio")
                return None
            
            logger.info(f"gTTS generated {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _edge_tts(self, text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
        """Generate speech using edge-tts (FREE, no API key)."""
        try:
            # Validate text
            if not text or not text.strip():
                logger.warning("Empty text provided to TTS")
                return None
            
            # Clean text (remove special characters that might cause issues)
            text = text.strip()
            
            # Use default voice if not specified (professional female voice)
            if not voice_id:
                voice_id = "en-US-AriaNeural"  # Professional female voice
            
            # Generate audio with edge-tts - try multiple methods
            async def _generate():
                # Method 1: Try using stream() first (most reliable)
                try:
                    logger.info(f"Generating audio with voice: {voice_id}, text length: {len(text)}")
                    communicate = edge_tts.Communicate(text, voice_id)
                    audio_data = b""
                    chunk_count = 0
                    
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                            chunk_count += 1
                        elif chunk["type"] == "metadata":
                            # Log metadata for debugging
                            logger.debug(f"Metadata: {chunk}")
                    
                    if len(audio_data) > 0:
                        logger.info(f"Stream method worked: {len(audio_data)} bytes from {chunk_count} chunks")
                        return audio_data
                    else:
                        logger.warning("Stream method returned no audio chunks")
                except Exception as stream_error:
                    logger.warning(f"Stream method failed: {stream_error}")
                
                # Method 2: Try using save() to BytesIO
                try:
                    logger.info("Trying save() method as fallback...")
                    audio_buffer = io.BytesIO()
                    communicate = edge_tts.Communicate(text, voice_id)
                    await communicate.save(audio_buffer)
                    audio_data = audio_buffer.getvalue()
                    
                    if len(audio_data) > 0:
                        logger.info(f"Save method worked: {len(audio_data)} bytes")
                        return audio_data
                    else:
                        logger.warning("Save method returned no audio")
                except Exception as save_error:
                    logger.warning(f"Save method failed: {save_error}")
                
                # Method 3: Try with temp file
                try:
                    logger.info("Trying temp file method as last resort...")
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    communicate = edge_tts.Communicate(text, voice_id)
                    await communicate.save(temp_path)
                    
                    # Read the file
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    # Clean up
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    if len(audio_data) > 0:
                        logger.info(f"Temp file method worked: {len(audio_data)} bytes")
                        return audio_data
                except Exception as temp_error:
                    logger.warning(f"Temp file method failed: {temp_error}")
                
                logger.error(f"All methods failed. Voice: {voice_id}, Text: {text[:50]}...")
                return None
            
            # Run async function
            audio_bytes = asyncio.run(_generate())
            
            if audio_bytes is None or len(audio_bytes) == 0:
                # Try with a different voice as fallback
                logger.info(f"Trying fallback voice: en-US-JennyNeural")
                async def _generate_fallback():
                    try:
                        audio_buffer = io.BytesIO()
                        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
                        await communicate.save(audio_buffer)
                        audio_data = audio_buffer.getvalue()
                        return audio_data if len(audio_data) > 0 else None
                    except Exception as e:
                        logger.error(f"Fallback voice also failed: {e}")
                        return None
                
                audio_bytes = asyncio.run(_generate_fallback())
            
            return audio_bytes
        except Exception as e:
            logger.error(f"edge-tts error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_audio_file(self, audio_bytes: bytes, filename: Optional[str] = None) -> Optional[str]:
        """
        Save audio bytes to a temporary file.
        
        Args:
            audio_bytes: Audio data as bytes
            filename: Optional filename (will create temp file if not provided)
        
        Returns:
            Path to saved audio file
        """
        try:
            if filename:
                filepath = Path(tempfile.gettempdir()) / filename
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                filepath = Path(temp_file.name)
                temp_file.close()
            
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None


# Singleton instance
_tts_service = None


def get_tts_service() -> TTSService:
    """Get or create TTS service instance (singleton)."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service

