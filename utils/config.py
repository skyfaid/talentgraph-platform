"""
Configuration file for AI Interviewer System
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EMODB_PATH = DATA_DIR / "EmoDB"
RAVDESS_PATH = DATA_DIR / "RAVDESS"
INTERVIEW_LOGS_PATH = DATA_DIR / "interview_logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
INTERVIEW_LOGS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Names
MISTRAL_MODEL = "mistral"  # Ollama model name
SENTIMENT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Interview Settings
MIN_QUESTIONS = 5
MAX_QUESTIONS = 10
TARGET_DURATION_MINUTES = 7

# Scoring Weights (TUNE THESE based on your needs)
SCORING_WEIGHTS = {
    "motivation": 0.30,      # Increased - more important for culture fit
    "experience": 0.35,      # Increased - most critical factor
    "logistics": 0.10,       # Decreased - less critical for initial screening
    "red_flags": 0.15,       # Decreased penalty weight
    "confidence": 0.10       # Voice confidence indicator
}

# Minimum word count for valid answers
MIN_ANSWER_LENGTH = 8  # Reduced from implicit 10

# Red Flag Detection Thresholds
RED_FLAG_SEVERITY = {
    'critical': 3.0,    # Deduct 3 points
    'major': 2.0,       # Deduct 2 points  
    'minor': 1.0        # Deduct 1 point
}

# Audio Settings
SAMPLE_RATE = 16000
AUDIO_DURATION_LIMIT = 60  # Max seconds per answer

# Emotion Labels (EmoDB + RAVDESS combined)
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

# Red Flag Keywords
RED_FLAG_KEYWORDS = [
    "i don't know",
    "not sure",
    "maybe",
    "i guess",
    "whatever",
    "don't care",
    "fired",
    "lawsuit",
    "conflict with"
]

# Positive Motivation Keywords
MOTIVATION_KEYWORDS = [
    "passionate",
    "excited",
    "interested",
    "motivated",
    "eager",
    "enthusiastic",
    "looking forward",
    "opportunity",
    "growth",
    "learn",
    "challenge"
]

# Experience Keywords (will be dynamically matched with job requirements)
SKILL_KEYWORDS = [
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "data science",
    "backend",
    "frontend",
    "full stack",
    "devops",
    "cloud",
    "aws",
    "azure",
    "docker",
    "kubernetes"
]
