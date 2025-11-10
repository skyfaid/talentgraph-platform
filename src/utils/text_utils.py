"""
Text processing utilities for resume cleaning and preparation.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException

# Download NLTK resources (only once)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
_stop_words = None


def get_stopwords():
    """Get stopwords set, initializing if needed."""
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
    return _stop_words


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'\d+', ' <NUM> ', text)  # replace numbers with token
    text = re.sub(r'[^a-zA-Z0-9\s<>]', '', text)  # remove punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with stopwords removed
    """
    stop_words = get_stopwords()
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])


def lemmatize_text(text: str) -> str:
    """
    Lemmatize words in text.
    
    Args:
        text: Text to lemmatize
        
    Returns:
        Lemmatized text
    """
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words])


def is_english(text: str) -> bool:
    """
    Check if text is in English.
    
    Args:
        text: Text to check
        
    Returns:
        True if text is English, False otherwise
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def prepare_resume_text(text: str, head_chars: int = 6000, tail_chars: int = 3000) -> str:
    """
    Prepare resume text for LLM processing by keeping head and tail.
    
    Args:
        text: Full resume text
        head_chars: Number of characters to keep from start
        tail_chars: Number of characters to keep from end
        
    Returns:
        Prepared resume text
    """
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n...\n" + text[-tail_chars:]


def extract_score(evaluation_text: str) -> float:
    """
    Extract numeric score from LLM evaluation text.
    
    Args:
        evaluation_text: LLM evaluation output
        
    Returns:
        Score as float (0-10), or 0.0 if not found
    """
    match = re.search(r'Score[:\s]+(\d+(?:\.\d+)?)/10', evaluation_text)
    if match:
        return float(match.group(1))
    return 0.0

