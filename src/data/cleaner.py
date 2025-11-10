"""
Data cleaning utilities for resume text processing.
"""
import pandas as pd
from typing import Optional
from ..utils.text_utils import (
    clean_text,
    remove_stopwords,
    lemmatize_text,
    is_english
)
from ..utils.config import MIN_RESUME_LENGTH


def clean_resume_dataframe(
    df: pd.DataFrame,
    text_column: str,
    category_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean resume text in a DataFrame.
    
    Args:
        df: DataFrame containing resume data
        text_column: Name of column containing resume text
        category_column: Optional name of category column
        
    Returns:
        DataFrame with cleaned text in 'clean_text' column
    """
    print("🧹 Cleaning text data...")
    
    # Create clean_text column
    df = df.copy()
    df['clean_text'] = df[text_column].fillna('').astype(str)
    
    # Apply cleaning steps
    df['clean_text'] = df['clean_text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(remove_stopwords)
    df['clean_text'] = df['clean_text'].apply(lemmatize_text)
    
    # Filter English resumes
    df = df[df['clean_text'].apply(is_english)]
    
    # Remove duplicates
    df.drop_duplicates(subset=['clean_text'], inplace=True)
    
    # Remove very short resumes
    df = df[df['clean_text'].str.len() > MIN_RESUME_LENGTH]
    
    # Add tokenized version
    df['tokens'] = df['clean_text'].apply(lambda x: x.split())
    
    print("✅ Data cleaning complete!")
    return df


def get_text_statistics(df: pd.DataFrame, text_column: str = 'clean_text') -> dict:
    """
    Get statistics about resume text data.
    
    Args:
        df: DataFrame with resume data
        text_column: Name of column containing text
        
    Returns:
        Dictionary with statistics
    """
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df['char_count'] = df[text_column].apply(len)
    
    stats = {
        'total_resumes': len(df),
        'avg_word_count': df['word_count'].mean(),
        'avg_char_count': df['char_count'].mean(),
        'min_words': df['word_count'].min(),
        'max_words': df['word_count'].max(),
        'min_chars': df['char_count'].min(),
        'max_chars': df['char_count'].max(),
    }
    
    # Count unique words
    all_text = ' '.join(df[text_column].tolist())
    stats['unique_words'] = len(set(all_text.split()))
    
    return stats

