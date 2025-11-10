# 🚀 AI-Powered CV Ranking System - Hybrid RAG + LLM

An advanced, universal CV ranking system that combines Retrieval-Augmented Generation (RAG) with Large Language Models (LLM) to intelligently rank resumes across all job categories.

## 🌟 Key Features

✅ **Universal Job Support** - Works for ALL job categories (tech, marketing, healthcare, finance, etc.)  
✅ **Hybrid RAG + LLM Architecture** - Combines vector search with intelligent LLM analysis  
✅ **Role-Based Prompting** - LLM acts as HR expert with few-shot examples  
✅ **Multi-Dataset Support** - Processes and combines multiple resume datasets  
✅ **Intelligent Text Cleaning** - Advanced NLP preprocessing for optimal analysis  
✅ **Hybrid Scoring** - 40% vector similarity + 60% LLM analysis for balanced ranking  

## 🏗️ System Architecture

### Frameworks & Technologies
- **LangChain**: Orchestration, chains, prompts, and vectorstore integration
- **Sentence-Transformers**: Embedding generation using `all-MiniLM-L6-v2`
- **ChromaDB**: Persistent vector storage
- **Groq Cloud**: LLM inference via `llama-4.1-17b-instant`
- **NLTK**: Text preprocessing and cleaning

### RAG Pipeline
1. **Retrieval Phase** - Semantic search using resume embeddings
2. **Augmentation Phase** - LLM enrichment for skills, experience, education extraction
3. **Generation Phase** - Hybrid scoring and final ranking with explanations

## 📊 Dataset Information

The system processes and combines two major resume datasets:
- **Resume.csv**: 2,484 resumes across multiple categories
- **UpdatedResumeDataSet.csv**: 962 professionally labeled resumes

**Total Processed**: 2,645 cleaned English resumes across 48 unique job categories

### Top Categories
- HR, Information Technology, Business Development
- Advocate, Chef, Engineering, Accountant
- Fitness, Finance, Aviation

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Groq API key

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/cv-ranker.git
cd cv-ranker

# Install required packages
pip install -r requirements.txt
