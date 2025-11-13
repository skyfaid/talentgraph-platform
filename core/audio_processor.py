"""
Audio Processing Module
Handles speech-to-text, TTS, and audio feature extraction
FIXED: TTS now works for all questions, not just the first
"""
import whisper
import librosa
import numpy as np
import pyttsx3
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple
import tempfile
import time

from utils.config import WHISPER_MODEL, SAMPLE_RATE, DATA_DIR


class AudioProcessor:
    def __init__(self):
        """Initialize audio processing components"""
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        
        # Don't initialize TTS engine here - we'll create fresh ones
        self.tts_engine = None
        
        # Audio recording settings
        self.sample_rate = SAMPLE_RATE
        self.recording = []
        self.is_recording = False
        
        # Create temp directory for audio files
        self.temp_dir = Path(DATA_DIR) / "temp_audio"
        # Ensure directory exists with proper permissions
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Convert to absolute path
        self.temp_dir = self.temp_dir.absolute()
        
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Convert speech to text using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Convert to Path object for proper path handling
            audio_file = Path(audio_path)
            
            if not audio_file.exists():
                return ""
            
            # Check file size
            if audio_file.stat().st_size == 0:
                return ""
            
            # Load and transcribe audio using librosa for consistent loading
            audio_data, sr = librosa.load(str(audio_file.absolute()), sr=16000, mono=True)
            result = self.whisper_model.transcribe(audio_data)
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def speak(self, text: str, save_path: str = None):
        """
        Convert text to speech with proper engine cleanup
        
        Args:
            text: Text to speak
            save_path: Optional path to save audio file
        """
        try:
            # Create a fresh engine for each call to avoid stuck state
            engine = pyttsx3.init()
            
            # Configure engine
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Get available voices
            voices = engine.getProperty('voices')
            
            # Try to set a better voice if available
            if len(voices) > 1:
                # Prefer second voice if available (often better quality)
                engine.setProperty('voice', voices[1].id)
            
            if save_path:
                engine.save_to_file(text, save_path)
                engine.runAndWait()
            else:
                engine.say(text)
                engine.runAndWait()
            
            # Important: Stop and delete the engine
            engine.stop()
            del engine
            
            # Small delay to ensure cleanup
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            # Make sure to clean up even on error
            try:
                if 'engine' in locals():
                    engine.stop()
                    del engine
            except:
                pass
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract prosodic and acoustic features from audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            # Extract features
            features = {}
            
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            features['pitch_mean'] = np.mean(pitch_values) if pitch_values else 0
            features['pitch_std'] = np.std(pitch_values) if pitch_values else 0
            
            # 3. Energy/Intensity
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # 4. Zero Crossing Rate (speech clarity)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            # 5. Speaking rate (duration analysis)
            duration = librosa.get_duration(y=y, sr=sr)
            features['duration'] = duration
            
            # 6. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {}
    
    def analyze_pauses(self, audio_path: str, threshold: float = 0.02) -> Dict[str, float]:
        """
        Analyze pauses and hesitations in speech
        
        Args:
            audio_path: Path to audio file
            threshold: Energy threshold for silence detection
            
        Returns:
            Pause statistics
        """
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            # Get RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Find silent frames
            silent_frames = rms < threshold
            
            # Count pauses
            pause_count = 0
            pause_durations = []
            in_pause = False
            pause_start = 0
            
            frame_duration = len(y) / sr / len(rms)
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    in_pause = True
                    pause_start = i
                elif not is_silent and in_pause:
                    in_pause = False
                    pause_duration = (i - pause_start) * frame_duration
                    if pause_duration > 0.3:  # Only count pauses > 0.3s
                        pause_count += 1
                        pause_durations.append(pause_duration)
            
            return {
                'pause_count': pause_count,
                'avg_pause_duration': np.mean(pause_durations) if pause_durations else 0,
                'total_pause_time': np.sum(pause_durations) if pause_durations else 0
            }
            
        except Exception as e:
            print(f"Error analyzing pauses: {e}")
            return {'pause_count': 0, 'avg_pause_duration': 0, 'total_pause_time': 0}
    
    def record_audio(self, duration: int = None, max_duration: int = 20) -> str:
        """
        Record audio from microphone
        
        Args:
            duration: Fixed duration in seconds (None for manual stop)
            max_duration: Maximum recording duration
            
        Returns:
            Path to saved audio file
        """
        print("\n🎤 Recording... Press ENTER to stop (or wait for auto-stop)")
        
        self.recording = []
        self.is_recording = True
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            if self.is_recording:
                self.recording.append(indata.copy())
        
        try:
            # Configure and start recording stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                dtype='float32',
                blocksize=2048,  # Smaller blocks for better responsiveness
                latency='low'    # Reduce latency
            )
            
            with stream:  # Use context manager to ensure proper cleanup
                print("Recording started...")
                
                if duration:
                    time.sleep(duration)
                    self.is_recording = False
                else:
                    # Manual stop with Enter key
                    import threading
                    
                    def wait_for_enter():
                        input()
                        self.is_recording = False
                    
                    enter_thread = threading.Thread(target=wait_for_enter)
                    enter_thread.daemon = True
                    enter_thread.start()
                    
                    # Wait for either Enter or max duration
                    start_time = time.time()
                    while self.is_recording and (time.time() - start_time) < max_duration:
                        time.sleep(0.1)
                    
                    self.is_recording = False
                    
        except Exception as e:
            print(f"⚠️ Error during recording: {e}")
            self.is_recording = False
            return None
        
        stream.stop()
        stream.close()
        
        print("✓ Recording stopped")
        
        # Convert recording to numpy array
        if not self.recording:
            print("Warning: No audio recorded")
            return None
        
        audio_data = np.concatenate(self.recording, axis=0)
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file
        timestamp = int(time.time())
        audio_path = self.temp_dir / f"recording_{timestamp}.wav"
        
        try:
            # Save audio file
            abs_path = str(audio_path.absolute())
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(abs_path, audio_data, self.sample_rate)
            
            if audio_path.exists() and audio_path.stat().st_size > 0:
                return abs_path
            return None
            
        except Exception as e:
            print(f"⚠️ Error saving audio: {e}")
            if audio_path.exists():
                try:
                    audio_path.unlink()  # Clean up partial file
                except:
                    pass
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_speaking_rate(self, audio_path: str, transcript: str) -> float:
        """
        Calculate words per minute
        
        Args:
            audio_path: Path to audio file
            transcript: Transcribed text
            
        Returns:
            Speaking rate (words per minute)
        """
        try:
            audio_file = Path(audio_path)
            if not audio_file.exists():
                print(f"Error: Audio file not found at {audio_path}")
                return 0
                
            y, sr = librosa.load(str(audio_file.absolute()), sr=SAMPLE_RATE)
            duration_minutes = librosa.get_duration(y=y, sr=sr) / 60
            word_count = len(transcript.split())
            
            return word_count / duration_minutes if duration_minutes > 0 else 0
            
        except Exception as e:
            print(f"Error calculating speaking rate: {e}")
            return 0