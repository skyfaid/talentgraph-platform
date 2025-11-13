"""
Video Processing Module
Handles video capture and real-time facial emotion detection
"""
import cv2
import numpy as np
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, List
from fer.fer import FER 

from utils.config import DATA_DIR, EMOTION_LABELS


class VideoProcessor:
    def __init__(self):
        """Initialize video processing components"""
        print("Initializing facial emotion detector...")
        self.detector = FER(mtcnn=True)
        
        # Video recording settings
        self.is_recording = False
        self.recording_thread = None
        self.video_writer = None
        self.emotion_history = []
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Emotion detection interval (seconds)
        self.detection_interval = 2.0
        self.last_detection_time = 0
        
        # Video capture
        self.cap = None
        self.current_video_path = None
        
        # Create temp directory for video files
        self.temp_dir = Path(DATA_DIR) / "temp_video"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_frame(self, frame):
        """Enhance frame for better emotion detection"""
        if frame is None:
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # Convert back to 3-channel (FER expects color)
            processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return processed
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return frame
    
    def detect_emotion_from_frame(self, frame) -> Optional[Dict]:
        """
        Detect emotion from a single frame
        
        Returns:
            Dictionary with emotion scores or None if no face detected
        """
        if frame is None:
            return None
            
        try:
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            if processed is None:
                return None
            
            # Detect emotions
            detections = self.detector.detect_emotions(processed)
            
            if not detections:
                return None
            
            # Pick the largest face (most stable)
            detections.sort(
                key=lambda d: d.get('box', [0, 0, 0, 0])[2] * d.get('box', [0, 0, 0, 0])[3],
                reverse=True
            )
            
            emotions = detections[0]['emotions']
            
            # Normalize emotion names to match our system
            normalized = {
                'angry': emotions.get('angry', 0),
                'disgust': emotions.get('disgust', 0),
                'fear': emotions.get('fear', 0),
                'happy': emotions.get('happy', 0),
                'sad': emotions.get('sad', 0),
                'surprise': emotions.get('surprise', 0),
                'neutral': emotions.get('neutral', 0)
            }
            
            return normalized
            
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return None
    
    def _recording_loop(self, output_path: str, fps: int = 15):
        """Internal recording loop (runs in thread)"""
        try:
            # Open video capture
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                self.is_recording = False
                return
            
            # Get frame dimensions
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            print(f"📹 Recording started: {output_path}")
            
            frame_delay = 1.0 / fps
            
            while self.is_recording:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Warning: Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Write frame to video
                self.video_writer.write(frame)
                
                # Update current frame (thread-safe)
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Detect emotion at intervals
                current_time = time.time()
                if current_time - self.last_detection_time >= self.detection_interval:
                    emotion_result = self.detect_emotion_from_frame(frame)
                    
                    if emotion_result:
                        self.emotion_history.append({
                            'timestamp': current_time,
                            'emotions': emotion_result
                        })
                    
                    self.last_detection_time = current_time
                
                # Control frame rate
                time.sleep(frame_delay)
            
        except Exception as e:
            print(f"Error in recording loop: {e}")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            print("✓ Video recording stopped")
    
    def start_recording(self, output_path: str = None, fps: int = 15):
        """
        Start video recording in background thread
        
        Args:
            output_path: Path to save video file
            fps: Frames per second
        """
        if self.is_recording:
            print("Warning: Already recording")
            return
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(self.temp_dir / f"interview_{timestamp}.mp4")
        
        self.current_video_path = output_path
        self.is_recording = True
        self.emotion_history = []
        self.last_detection_time = time.time()
        
        # Start recording in background thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(output_path, fps),
            daemon=True
        )
        self.recording_thread.start()
        
        # Wait a moment for camera to initialize
        time.sleep(1.0)
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Wait for thread to finish (with timeout)
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        self.recording_thread = None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame (thread-safe)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def show_preview_window(self, duration: float = 5.0):
        """
        Show live preview window for specified duration
        
        Args:
            duration: How long to show preview (seconds)
        """
        if not self.is_recording:
            print("Error: Not recording")
            return
        
        print(f"📹 Showing preview for {duration} seconds...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                frame = self.get_current_frame()
                
                if frame is not None:
                    # Add recording indicator
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show emotion if available
                    if self.emotion_history:
                        latest = self.emotion_history[-1]['emotions']
                        dominant = max(latest.items(), key=lambda x: x[1])
                        cv2.putText(frame, f"{dominant[0]}: {dominant[1]:.2f}", 
                                   (10, frame.shape[0] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('AI Interview - Video Mode', frame)
                
                # Check for 'q' key to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)  # ~30 FPS display
        
        except Exception as e:
            print(f"Error showing preview: {e}")
        finally:
            cv2.destroyAllWindows()
    
    def get_aggregated_emotions(self) -> Dict[str, float]:
        """
        Aggregate emotions from entire recording
        
        Returns:
            Dictionary with average emotion scores
        """
        if not self.emotion_history:
            return {
                'neutral': 0.5,
                'happy': 0,
                'sad': 0,
                'angry': 0,
                'fear': 0,
                'disgust': 0,
                'surprise': 0
            }
        
        # Average all emotion scores
        aggregated = defaultdict(list)
        
        for entry in self.emotion_history:
            for emotion, score in entry['emotions'].items():
                aggregated[emotion].append(score)
        
        # Calculate averages
        result = {}
        for emotion, scores in aggregated.items():
            result[emotion] = sum(scores) / len(scores) if scores else 0
        
        return result
    
    def get_emotion_timeline(self) -> List[Dict]:
        """
        Get emotion changes over time
        
        Returns:
            List of emotion snapshots with timestamps
        """
        return self.emotion_history.copy()
    
    def get_dominant_emotions(self, top_n: int = 3) -> List[tuple]:
        """
        Get top N dominant emotions across interview
        
        Returns:
            List of (emotion, average_score) tuples
        """
        aggregated = self.get_aggregated_emotions()
        sorted_emotions = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_n]
    
    def cleanup_temp_files(self):
        """Clean up temporary video files"""
        import shutil
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                print("✓ Temp video files cleaned up")
            except Exception as e:
                print(f"Warning: Could not clean temp files: {e}")