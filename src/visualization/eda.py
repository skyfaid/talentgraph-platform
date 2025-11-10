"""
Exploratory Data Analysis and visualization utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional


def plot_category_distribution(df: pd.DataFrame, category_column: str = 'category', top_n: int = 20):
    """
    Plot distribution of resume categories.
    
    Args:
        df: DataFrame with resume data
        category_column: Name of category column
        top_n: Number of top categories to show
    """
    if category_column not in df.columns:
        print(f"⚠️ Column '{category_column}' not found in DataFrame")
        return
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    top_categories = df[category_column].value_counts().head(top_n)
    sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
    
    plt.title(f"Top {top_n} Resume Categories by Count")
    plt.xlabel("Number of Resumes")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()


def plot_resume_length_distributions(df: pd.DataFrame, word_count_column: str = 'word_count', char_count_column: str = 'char_count'):
    """
    Plot distributions of resume word and character counts.
    
    Args:
        df: DataFrame with resume data
        word_count_column: Name of word count column
        char_count_column: Name of character count column
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))
    
    # Word count histogram
    if word_count_column in df.columns:
        plt.subplot(1, 2, 1)
        sns.histplot(df[word_count_column], bins=50, kde=True, color='skyblue')
        plt.title("Distribution of Resume Word Counts")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Resumes")
    
    # Character count histogram
    if char_count_column in df.columns:
        plt.subplot(1, 2, 2)
        sns.histplot(df[char_count_column], bins=50, kde=True, color='salmon')
        plt.title("Distribution of Resume Character Counts")
        plt.xlabel("Character Count")
        plt.ylabel("Number of Resumes")
    
    plt.tight_layout()
    plt.show()


def plot_avg_words_per_category(df: pd.DataFrame, category_column: str = 'category', word_count_column: str = 'word_count', top_n: int = 20):
    """
    Plot average word count per category.
    
    Args:
        df: DataFrame with resume data
        category_column: Name of category column
        word_count_column: Name of word count column
        top_n: Number of top categories to show
    """
    if category_column not in df.columns or word_count_column not in df.columns:
        print(f"⚠️ Required columns not found in DataFrame")
        return
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    avg_words_per_category = (
        df.groupby(category_column)[word_count_column]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    
    sns.barplot(x=avg_words_per_category.values, y=avg_words_per_category.index, palette="magma")
    plt.title(f"Average Word Count per Category (Top {top_n} Categories)")
    plt.xlabel("Average Word Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()

