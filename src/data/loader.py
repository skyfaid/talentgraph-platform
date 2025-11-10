"""
Data loading utilities for CSV resume datasets.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from ..utils.config import DATA_DIR


def load_resume_csv(file_path: Optional[Path] = None, column_name: str = "Resume_str") -> pd.DataFrame:
    """
    Load resume data from a CSV file.
    
    Args:
        file_path: Path to CSV file. If None, uses default Resume.csv
        column_name: Name of the column containing resume text
        
    Returns:
        DataFrame with resume data
    """
    if file_path is None:
        file_path = DATA_DIR / "Resume.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} resumes from {file_path.name}")
    return df


def load_all_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both resume datasets.
    
    Returns:
        Tuple of (df1, df2) DataFrames
    """
    df1 = load_resume_csv(DATA_DIR / "Resume.csv", column_name="Resume_str")
    df2 = load_resume_csv(DATA_DIR / "UpdatedResumeDataSet.csv", column_name="Resume")
    return df1, df2

