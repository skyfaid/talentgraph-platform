"""
CV Ranking System - Main Script

Simple script to run the CV ranking pipeline.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import (
    load_all_datasets,
    clean_resume_dataframe,
    combine_resume_datasets,
    resumes_to_dataframe
)
from src.embeddings import (
    create_embeddings,
    create_vectorstore,
    create_documents_from_resumes
)
from src.llm import initialize_llm, create_evaluation_chain
from src.ranker import CVRanker


def main():
    """Main execution - runs the full CV ranking pipeline."""
    print("🚀 CV Ranking System\n")
    
    # Step 1: Load and clean data
    print("📂 Loading datasets...")
    df1, df2 = load_all_datasets()
    
    print("🧹 Cleaning data...")
    df1_clean = clean_resume_dataframe(df1, text_column="Resume_str", category_column="Category")
    df2_clean = clean_resume_dataframe(df2, text_column="Resume", category_column="Category")
    
    print("🔄 Combining datasets...")
    all_resumes = combine_resume_datasets(
        df1_clean, df2_clean,
        id_column1="ID",
        category_column="Category",
        text_column="clean_text"
    )
    print(f"✅ Total resumes: {len(all_resumes)}\n")
    
    # Step 2: Create embeddings and vectorstore
    print("🔬 Creating embeddings...")
    documents = create_documents_from_resumes(all_resumes)
    embeddings = create_embeddings()
    vectorstore = create_vectorstore(documents, embeddings)
    print("✅ Vectorstore created\n")
    
    # Step 3: Initialize LLM
    print("🤖 Initializing LLM...")
    try:
        llm = initialize_llm()
        evaluation_chain = create_evaluation_chain(llm)
        print("✅ LLM ready\n")
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        print("💡 Set GROQ_API_KEY environment variable")
        return
    
    # Step 4: Create ranker and test
    print("🎯 Creating ranker...")
    ranker = CVRanker(vectorstore, evaluation_chain)
    
    # Test ranking
    job_description = "Senior Data Engineer with Python, SQL, AWS; 5+ years; leadership a plus."
    print(f"\n📋 Ranking candidates for:\n   {job_description}\n")
    
    results = ranker.rank_resumes(job_description, top_k=5)
    
    # Display results
    print("🏆 Top 5 Candidates:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r['final_score']}/10")
        print(f"   LLM: {r['llm_score']}/10 | Similarity: {r['semantic_similarity']:.3f}")
        print(f"   Category: {r['meta'].get('category', 'Unknown')}")
        print(f"   Preview: {r['preview'][:100]}...")
        print()
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
