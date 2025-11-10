"""
Utilities for combining multiple resume datasets.
"""
import pandas as pd
from typing import List, Dict, Optional
from ..utils.candidate_generator import generate_candidate_info_deterministic


def combine_resume_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_column1: str = "ID",
    id_column2: Optional[str] = None,
    category_column: str = "Category",
    text_column: str = "clean_text",
    source1: str = "Resume.csv",
    source2: str = "UpdatedResumeDataSet.csv"
) -> List[Dict]:
    """
    Combine two resume DataFrames into a unified list of dictionaries.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        id_column1: ID column name in df1
        id_column2: ID column name in df2 (if None, uses index)
        category_column: Category column name
        text_column: Text column name
        source1: Source name for df1
        source2: Source name for df2
        
    Returns:
        List of resume dictionaries
    """
    print("🔄 Combining datasets...")
    
    all_resumes = []
    
    # Add resumes from first file
    for idx, row in df1.iterrows():
        resume_id = f"resume1_{row[id_column1]}" if id_column1 in df1.columns else f"resume1_{idx}"
        
        # Get name and email from CSV if available, otherwise generate
        if 'name' in df1.columns and 'email' in df1.columns:
            name = row.get('name', 'Unknown Candidate')
            email = row.get('email', 'unknown@example.com')
        else:
            # Generate if not in CSV
            candidate_info = generate_candidate_info_deterministic(resume_id)
            name = candidate_info['name']
            email = candidate_info['email']
        
        all_resumes.append({
            'id': resume_id,
            'text': row[text_column],
            'category': row[category_column] if category_column in df1.columns else 'Unknown',
            'source': source1,
            'name': name,
            'email': email
        })
    
    # Add resumes from second file
    for idx, row in df2.iterrows():
        if id_column2 and id_column2 in df2.columns:
            resume_id = f"resume2_{row[id_column2]}"
        else:
            resume_id = f"resume2_{idx}"
        
        # Get name and email from CSV if available, otherwise generate
        if 'name' in df2.columns and 'email' in df2.columns:
            name = row.get('name', 'Unknown Candidate')
            email = row.get('email', 'unknown@example.com')
        else:
            # Generate if not in CSV
            candidate_info = generate_candidate_info_deterministic(resume_id)
            name = candidate_info['name']
            email = candidate_info['email']
        
        all_resumes.append({
            'id': resume_id,
            'text': row[text_column],
            'category': row[category_column] if category_column in df2.columns else 'Unknown',
            'source': source2,
            'name': name,
            'email': email
        })
    
    print(f"✅ Combined datasets: {len(all_resumes)} total resumes")
    return all_resumes


def resumes_to_dataframe(resumes: List[Dict]) -> pd.DataFrame:
    """
    Convert list of resume dictionaries to DataFrame.
    
    Args:
        resumes: List of resume dictionaries
        
    Returns:
        DataFrame with resume data
    """
    df = pd.DataFrame(resumes)
    
    # Add word count and character length columns
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(len)
    
    return df


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about the combined dataset.
    
    Args:
        df: DataFrame with combined resumes
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_resumes': len(df),
        'unique_categories': df['category'].nunique() if 'category' in df.columns else 0,
        'avg_words': df['word_count'].mean() if 'word_count' in df.columns else 0,
        'avg_chars': df['char_count'].mean() if 'char_count' in df.columns else 0,
    }
    
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        stats['top_categories'] = category_counts.head(10).to_dict()
    
    return stats


def print_dataset_summary(df: pd.DataFrame):
    """
    Print a summary of the dataset.
    
    Args:
        df: DataFrame with combined resumes
    """
    print(f"✅ Combined datasets: {len(df)} total resumes")
    
    if 'category' in df.columns:
        print(f"📊 Number of unique categories: {df['category'].nunique()}")
        
        category_counts = df['category'].value_counts()
        category_percent = round(df['category'].value_counts(normalize=True) * 100, 2)
        
        print("\n🏆 Top 10 categories by count:")
        top_categories = category_counts.head(10)
        for cat, count in top_categories.items():
            pct = category_percent[cat]
            print(f"   {cat}: {count} resumes ({pct}%)")
        
        print("\n📊 Average resume statistics per category (top 10 categories):")
        for cat in top_categories.index:
            subset = df[df['category'] == cat]
            if 'word_count' in df.columns and 'char_count' in df.columns:
                avg_words = subset['word_count'].mean()
                avg_chars = subset['char_count'].mean()
                print(f"   {cat}: avg words = {avg_words:.0f}, avg chars = {avg_chars:.0f}")
    
    if 'word_count' in df.columns and 'char_count' in df.columns:
        print("\n📝 Overall dataset statistics:")
        print(f"   Total resumes: {len(df)}")
        print(f"   Unique categories: {df['category'].nunique() if 'category' in df.columns else 'N/A'}")
        print(f"   Avg words per resume: {df['word_count'].mean():.0f}")
        print(f"   Avg characters per resume: {df['char_count'].mean():.0f}")
        print(f"   Min words: {df['word_count'].min()}, Max words: {df['word_count'].max()}")
        print(f"   Min chars: {df['char_count'].min()}, Max chars: {df['char_count'].max()}")

