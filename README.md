# 🚀 CV Ranking System

AI-powered CV ranking system using hybrid RAG + LLM approach. Ranks resumes against job descriptions using semantic search and intelligent LLM analysis with explainable AI (XAI) features.

## 📋 What It Does

1. **Loads CV data** from CSV files (with candidate names and emails)
2. **Cleans and processes** resume text
3. **Creates embeddings** using sentence transformers
4. **Stores in vector database** (ChromaDB) for fast search
5. **Ranks candidates** using:
   - 30% semantic similarity (fast keyword/skill matching)
   - 70% LLM analysis (deep understanding of fit)
6. **Provides explanations** using SHAP, LIME, and rule-based analysis

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

You need a Groq API key for LLM functionality:

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_api_key_here"

# Windows CMD
set GROQ_API_KEY=your_api_key_here

# Linux/Mac
export GROQ_API_KEY="your_api_key_here"
```

**Option B: Edit config file**
Edit `src/utils/config.py` and set `GROQ_API_KEY` directly (not recommended for production).

Get your API key from: https://console.groq.com/

### 3. Prepare Data

Make sure your CSV files are in the `data/` folder:
- `data/Resume.csv` (with columns: `ID`, `Resume_str`, `Category`, `name`, `email`)
- `data/UpdatedResumeDataSet.csv` (with columns: `Resume`, `Category`, `name`, `email`)

**Note:** If your CSV files don't have `name` and `email` columns, the system will generate them automatically.

## ▶️ How to Run

### Option 1: Command Line Script (`main.py`)

Simple script for testing and development:

```bash
python main.py
```

This will:
1. Load and clean your resume data
2. Create embeddings and vectorstore
3. Initialize LLM
4. Rank candidates for a sample job description
5. Display top 5 candidates

**Customize:** Edit `main.py` and change the `job_description` variable.

### Option 2: FastAPI Server (`app.py`)

Production-ready REST API:

```bash
python app.py
```

Then access:
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/health
- **Rank Endpoint**: http://localhost:8000/rank

**Example Request:**
```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Data Engineer with Python, SQL, AWS; 5+ years; leadership a plus.",
    "top_k": 5,
    "include_explanations": true
  }'
```

## 📁 Project Structure

```
talentgraph-platform/
├── main.py                    # CLI script for testing ranking
├── app.py                     # FastAPI REST API server
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── data/                      # CSV data files
│   ├── Resume.csv            # Main resume dataset (2,484+ resumes)
│   └── UpdatedResumeDataSet.csv  # Additional dataset (962+ resumes)
│
├── chroma_db/                 # Vector database (auto-generated)
│   └── ...                    # ChromaDB files
│
└── src/                       # Source code
    ├── __init__.py
    │
    ├── api/                  # FastAPI REST API
    │   ├── __init__.py
    │   ├── models.py        # Pydantic models (requests/responses)
    │   └── service.py        # Service layer (initializes components)
    │
    ├── data/                 # Data loading and processing
    │   ├── __init__.py
    │   ├── loader.py        # Load CSV files
    │   ├── cleaner.py       # Clean and normalize text
    │   └── combiner.py      # Combine multiple datasets
    │
    ├── embeddings/           # Embedding and vector storage
    │   ├── __init__.py
    │   ├── embedder.py      # Create embeddings (HuggingFace)
    │   └── vectorstore.py   # ChromaDB vectorstore management
    │
    ├── llm/                  # LLM service (Groq Cloud)
    │   ├── __init__.py
    │   └── groq_service.py   # Groq LLM initialization and chains
    │
    ├── ranker/               # Ranking logic
    │   ├── __init__.py
    │   └── cv_ranker.py      # Hybrid ranking (semantic + LLM)
    │
    ├── xai/                  # Explainable AI
    │   ├── __init__.py
    │   ├── explainer.py      # Main explainer (rule-based + SHAP + LIME)
    │   ├── shap_explainer.py # SHAP feature importance
    │   └── lime_explainer.py # LIME text-level importance
    │
    ├── pdf/                  # PDF parsing (for future uploads)
    │   ├── __init__.py
    │   └── parser.py         # Extract text from PDF files
    │
    ├── utils/                # Utilities and configuration
    │   ├── __init__.py
    │   ├── config.py         # Configuration settings
    │   ├── logger.py         # Logging setup
    │   ├── text_utils.py     # Text processing utilities
    │   └── candidate_generator.py  # Generate candidate names/emails
    │
    ├── mlops/                # MLOps and experiment tracking
    │   ├── __init__.py
    │   ├── metrics.py        # Ranking metrics calculation
    │   └── mlflow_tracker.py # MLflow experiment tracking
    │
    └── visualization/        # Data visualization (EDA)
        ├── __init__.py
        └── eda.py            # Exploratory data analysis plots
```

## 📄 File Descriptions

### Main Files

#### `main.py`
**Purpose:** Command-line script for testing the ranking system.

**What it does:**
- Loads and cleans resume data from CSV files
- Creates embeddings and vectorstore
- Initializes Groq LLM
- Runs a sample ranking query
- Displays results in console

**Use when:** Testing, development, or quick ranking without API.

---

#### `app.py`
**Purpose:** FastAPI REST API server for production use.

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check (vectorstore, LLM status)
- `POST /rank` - Rank candidates against job description
- `POST /explain` - Get detailed explanation for a specific candidate
- `POST /upload` - Upload PDF CVs (future feature)

**Features:**
- Automatic service initialization on startup
- CORS enabled for frontend integration
- Swagger UI documentation at `/docs`
- XAI explanations (SHAP + LIME) when requested

**Use when:** Production deployment, API integration, or web frontend.

---

### Source Code Modules

#### `src/api/` - FastAPI REST API

**`models.py`**
- Pydantic models for API requests/responses
- `RankRequest` - Ranking request with job description
- `RankResponse` - Ranked candidates response
- `CandidateResult` - Individual candidate result
- `Explanation` - XAI explanation structure
- `SHAPAnalysis`, `LIMEAnalysis` - XAI analysis models

**`service.py`**
- `CVRankingService` - Service layer singleton
- Manages vectorstore, LLM, and ranker initialization
- Handles lazy loading and caching
- Auto-detects if vectorstore needs recreation

---

#### `src/data/` - Data Processing

**`loader.py`**
- `load_resume_csv()` - Load CSV files
- `load_all_datasets()` - Load both resume datasets
- Handles file paths and error checking

**`cleaner.py`**
- `clean_resume_dataframe()` - Clean and normalize text
- Removes HTML tags, normalizes whitespace
- Filters non-English resumes
- Handles missing values

**`combiner.py`**
- `combine_resume_datasets()` - Merge multiple datasets
- Generates unique IDs for each resume
- Reads candidate names/emails from CSV (or generates them)
- Creates unified resume dictionary format

---

#### `src/embeddings/` - Embedding & Vector Storage

**`embedder.py`**
- `create_embeddings()` - Initialize HuggingFace embeddings
- Uses `sentence-transformers/all-MiniLM-L6-v2` model
- Converts text to 384-dimensional vectors

**`vectorstore.py`**
- `create_vectorstore()` - Create ChromaDB vectorstore
- `load_vectorstore()` - Load existing vectorstore
- `create_documents_from_resumes()` - Convert resumes to LangChain Documents
- Handles persistence and metadata storage

---

#### `src/llm/` - LLM Service

**`groq_service.py`**
- `initialize_llm()` - Initialize Groq Cloud LLM
- `create_evaluation_chain()` - Create LLMChain for candidate evaluation
- Configures model parameters (temperature, top_p, seed)
- Handles API key validation

**Models used:**
- Default: `meta-llama/llama-4-scout-17b-16e-instruct`
- Configurable in `src/utils/config.py`

---

#### `src/ranker/` - Ranking Logic

**`cv_ranker.py`**
- `CVRanker` class - Main ranking engine
- **Stage 1:** Semantic search (fast vector similarity)
- **Stage 2:** LLM evaluation (deep analysis per candidate)
- **Stage 3:** Hybrid scoring (30% semantic + 70% LLM)
- Returns ranked candidates with scores and metadata

**Key methods:**
- `rank_resumes()` - Main ranking method
- `get_retriever()` - Get semantic search retriever

---

#### `src/xai/` - Explainable AI

**`explainer.py`**
- `RankingExplainer` - Main explainer class
- **Rule-based:** Skill matching, score breakdowns
- **SHAP integration:** Feature-level importance
- **LIME integration:** Word/phrase-level importance
- Extracts strengths, weaknesses, experience info

**`shap_explainer.py`**
- `SHAPExplainer` - SHAP feature importance
- Explains which skills contributed most to ranking
- Calculates skill importance scores

**`lime_explainer.py`**
- `LIMEExplainer` - LIME text-level importance
- Shows which words/phrases in resume influenced ranking
- Filters noise words, prioritizes tech keywords

---

#### `src/utils/` - Utilities

**`config.py`**
- All configuration settings
- Paths, model names, API keys
- Scoring weights, text processing settings
- **Edit this to customize the system**

**`logger.py`**
- `setup_logger()` - Configure logging
- Creates log files in `logs/` directory
- Different log levels for different modules

**`text_utils.py`**
- `prepare_resume_text()` - Prepare text for LLM (head + tail)
- `extract_score()` - Extract numeric score from LLM text
- Text processing utilities

**`candidate_generator.py`**
- `generate_candidate_name()` - Generate realistic names
- `generate_candidate_email()` - Generate emails
- `generate_candidate_info_deterministic()` - Deterministic generation (same ID = same name)

---

#### `src/pdf/` - PDF Processing

**`parser.py`**
- `extract_text_from_multiple_pdfs()` - Extract text from PDF files
- Used by `/upload` endpoint for PDF CV uploads
- Supports multiple PDF libraries (pdfplumber, PyPDF2)

---

#### `src/mlops/` - MLOps & Tracking

**`metrics.py`**
- `calculate_ranking_metrics()` - Calculate ranking performance metrics
- `calculate_category_consistency()` - Category matching metrics
- `log_evaluation_metrics()` - Log metrics to MLflow

**`mlflow_tracker.py`**
- `MLflowTracker` - MLflow experiment tracking
- Log parameters, metrics, artifacts
- Model versioning and registry
- Experiment management

---

#### `src/visualization/` - Data Visualization

**`eda.py`**
- `plot_category_distribution()` - Plot resume categories
- `plot_resume_length_distributions()` - Word/char count distributions
- `plot_avg_words_per_category()` - Average words per category
- Useful for understanding your dataset

---

## 🔧 How It Works

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CV RANKING SYSTEM ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────┐              ┌──────────────┐
        │ Resume.csv   │              │UpdatedResume │
        │ (2,484 CVs)  │              │DataSet.csv   │
        │              │              │ (962 CVs)    │
        └──────┬───────┘              └──────┬───────┘
               │                            │
               └────────────┬───────────────┘
                            ▼
              ┌─────────────────────────┐
              │  Data Loader & Cleaner   │
              │  - Load CSV files        │
              │  - Clean text            │
              │  - Normalize format      │
              │  - Filter English        │
              └────────────┬─────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Data Combiner          │
              │  - Merge datasets       │
              │  - Generate names/emails│
              │  - Create unified format │
              └────────────┬─────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Candidate Generator    │
              │  - Generate names       │
              │  - Generate emails      │
              │  (if not in CSV)        │
              └────────────┬─────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING & STORAGE LAYER                        │
└─────────────────────────────────────────────────────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  HuggingFace Embeddings │
              │  - Model: all-MiniLM-   │
              │    L6-v2                │
              │  - Convert text →       │
              │    384-dim vectors      │
              └────────────┬─────────────┘
                           ▼
              ┌─────────────────────────┐
              │  ChromaDB Vectorstore   │
              │  - Store embeddings     │
              │  - Fast similarity      │
              │    search               │
              │  - Persistent storage  │
              │  - Metadata: name,     │
              │    email, category      │
              └────────────┬─────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           RANKING LAYER                                 │
└─────────────────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌──────────────────┐              ┌──────────────────┐
│  Job Description │              │  User Request    │
│  (Input)         │              │  (API/CLI)       │
└────────┬─────────┘              └────────┬─────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   STAGE 1: Semantic Search   │
         │   - Vector similarity        │
         │   - Fast keyword matching    │
         │   - Retrieves 12 candidates  │
         │     (2x top_k, min 12)      │
         └──────────────┬──────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   STAGE 2: LLM Evaluation    │
         │   - Groq Cloud LLM           │
         │   - Deep analysis per        │
         │     candidate                │
         │   - Scores 0-10              │
         │   - Detailed evaluation text │
         └──────────────┬──────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   STAGE 3: Hybrid Scoring    │
         │   - 30% semantic similarity  │
         │   - 70% LLM score            │
         │   - Weighted combination     │
         │   - Final score (0-10)       │
         └──────────────┬──────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   STAGE 4: Ranking & Filter  │
         │   - Sort by final score      │
         │   - Return top_k candidates  │
         │   - Include metadata        │
         └──────────────┬──────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPLAINABLE AI (XAI) LAYER                       │
└─────────────────────────────────────────────────────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   XAI Explanation Engine    │
         │   (if requested)             │
         └──────────────┬───────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Rule-Based  │ │    SHAP     │ │    LIME    │
│ Analysis    │ │  Analysis   │ │  Analysis  │
│             │ │             │ │            │
│ - Skill     │ │ - Feature   │ │ - Word/    │
│   matching  │ │   importance│ │   phrase   │
│ - Score     │ │ - Skill     │ │   level    │
│   breakdown │ │   contrib.  │ │   import.  │
│ - Strengths │ │ - Top       │ │ - Section  │
│   /weakness │ │   skills    │ │   import.  │
└─────────────┘ └─────────────┘ └─────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   Combined Explanation       │
         │   - All three methods        │
         │   - Comprehensive insights   │
         └──────────────┬───────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                  │
└─────────────────────────────────────────────────────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   Ranked Candidates          │
         │   - Final scores             │
         │   - LLM evaluations          │
         │   - Candidate names/emails   │
         │   - Metadata                │
         │   - XAI explanations         │
         │     (optional)               │
         └──────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │   API Response / CLI Output   │
        │   - JSON format               │
        │   - Top K candidates          │
        │   - Complete explanations     │
        └───────────────────────────────┘
```

### Detailed Ranking Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE RANKING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Job Description
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: SEMANTIC SEARCH (Fast - ~100ms)                        │
│  ────────────────────────────────────────────────────────────── │
│  • Convert job description to embedding vector                   │
│  • Search ChromaDB for similar resumes                          │
│  • Use cosine similarity                                        │
│  • Retrieve top 12 candidates (2x top_k, min 12)              │
│                                                                  │
│  Output: [(resume_doc, similarity_score), ...]                  │
│          Example: [(doc1, 0.85), (doc2, 0.72), ...]            │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: LLM EVALUATION (Slower - ~2-5s per candidate)           │
│  ────────────────────────────────────────────────────────────── │
│  For each of 12 candidates:                                      │
│    • Prepare resume text (head + tail to fit tokens)            │
│    • Send to Groq LLM:                                          │
│        Input: Resume + Job Description                           │
│        Prompt: "Evaluate this candidate..."                      │
│    • LLM returns: "Score: 8/10. The candidate has..."           │
│    • Extract numeric score: 8.0                                 │
│                                                                  │
│  Output: [                                                        │
│    {candidate: doc1, llm_score: 8.0, evaluation: "..."},         │
│    {candidate: doc2, llm_score: 7.5, evaluation: "..."},       │
│    ...                                                           │
│  ]                                                               │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: HYBRID SCORING (Fast - ~1ms per candidate)             │
│  ────────────────────────────────────────────────────────────── │
│  For each candidate:                                             │
│    • Semantic similarity: 0.85 (from Step 1)                     │
│    • LLM score: 8.0 (from Step 2)                               │
│    • Calculate:                                                  │
│        Final = 0.3 × (0.85 × 10) + 0.7 × 8.0                    │
│             = 0.3 × 8.5 + 0.7 × 8.0                             │
│             = 2.55 + 5.6                                        │
│             = 8.15                                               │
│                                                                  │
│  Output: [                                                        │
│    {final_score: 8.15, llm_score: 8.0, semantic: 0.85, ...},   │
│    {final_score: 7.82, llm_score: 7.5, semantic: 0.72, ...},    │
│    ...                                                           │
│  ]                                                               │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: RANKING & FILTERING (Fast - ~1ms)                      │
│  ────────────────────────────────────────────────────────────── │
│  • Sort all candidates by final_score (descending)               │
│  • Take top K candidates (e.g., top 5)                          │
│  • Include metadata (name, email, category)                      │
│                                                                  │
│  Output: Top 5 ranked candidates                                │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: XAI EXPLANATIONS (Optional - ~1-2s per candidate)     │
│  ────────────────────────────────────────────────────────────── │
│  If include_explanations=true:                                   │
│    For each top candidate:                                      │
│      ┌────────────────────────────────────────┐                  │
│      │ Rule-Based Analysis                    │                  │
│      │ • Extract skills from LLM text        │                  │
│      │ • Calculate skill match rate          │                  │
│      │ • Extract strengths/weaknesses        │                  │
│      └────────────────────────────────────────┘                  │
│      ┌────────────────────────────────────────┐                  │
│      │ SHAP Analysis (if available)          │                  │
│      │ • Calculate skill importance          │                  │
│      │ • Show which skills boosted score     │                  │
│      └────────────────────────────────────────┘                  │
│      ┌────────────────────────────────────────┐                  │
│      │ LIME Analysis (if available)          │                  │
│      │ • Analyze word-level importance       │                  │
│      │ • Show which text parts mattered      │                  │
│      └────────────────────────────────────────┘                  │
│      • Combine all explanations                                │
│                                                                  │
│  Output: Complete explanation with all three methods            │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                                    │
│  ────────────────────────────────────────────────────────────── │
│  {                                                               │
│    "job_description": "...",                                    │
│    "top_k": 5,                                                   │
│    "candidates": [                                               │
│      {                                                           │
│        "final_score": 8.15,                                      │
│        "llm_score": 8.0,                                         │
│        "semantic_similarity": 0.85,                              │
│        "meta": {                                                  │
│          "id": "resume1_123",                                    │
│          "name": "John Smith",                                   │
│          "email": "john.smith456@gmail.com",                      │
│          "category": "INFORMATION-TECHNOLOGY"                     │
│        },                                                         │
│        "evaluation": "Score: 8/10...",                            │
│        "explanation": { ... }  // if requested                  │
│      },                                                           │
│      ...                                                          │
│    ]                                                              │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌──────────────┐
│   User/API   │
│   Request    │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI App (app.py)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  /rank   │  │ /explain │  │ /health  │  │ /upload  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼──────────────┼──────────────┼──────────────┼────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              CVRankingService (service.py)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Vectorstore  │  │     LLM      │  │   Ranker    │     │
│  │ (ChromaDB)   │  │   (Groq)     │  │  (Hybrid)   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│  Embeddings     │  │  Evaluation  │  │  Ranking     │
│  (HuggingFace)  │  │  Chain       │  │  Logic       │
│                 │  │  (LangChain) │  │              │
└─────────────────┘  └──────────────┘  └──────┬───────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  XAI Explainer   │
                                    │  (SHAP + LIME)   │
                                    └──────────────────┘
```

### Data Flow Diagram

```
┌─────────────┐
│  CSV Files  │
│  (data/)    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Data Processing Pipeline                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Loader  │→ │  Cleaner │→ │ Combiner │            │
│  └──────────┘  └──────────┘  └────┬─────┘            │
└─────────────────────────────────────┼───────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Resume Dicts     │
                            │ - text           │
                            │ - name           │
                            │ - email          │
                            │ - category       │
                            └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Embeddings      │
                            │  (384-dim)       │
                            └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  ChromaDB        │
                            │  Vectorstore     │
                            │  (Persistent)    │
                            └──────────────────┘
                                     │
                                     │ (On Request)
                                     ▼
                            ┌──────────────────┐
                            │  Ranking Query   │
                            │  Job Description│
                            └────────┬─────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
                    ▼                                  ▼
         ┌──────────────────┐              ┌──────────────────┐
         │ Semantic Search  │              │  LLM Evaluation   │
         │ (Vector Similar) │              │  (Deep Analysis)  │
         │ Score: 0-1       │              │  Score: 0-10     │
         └────────┬─────────┘              └────────┬─────────┘
                  │                                  │
                  └──────────────┬───────────────────┘
                                 ▼
                      ┌──────────────────┐
                      │  Hybrid Scoring  │
                      │  30% + 70%       │
                      └────────┬─────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  Ranked Results  │
                      │  + Explanations  │
                      └──────────────────┘
```

### Step-by-Step Process

1. **Data Loading** (`src/data/loader.py`)
   - Reads CSV files with resumes
   - Handles multiple datasets

2. **Text Cleaning** (`src/data/cleaner.py`)
   - Removes HTML tags, normalizes whitespace
   - Filters non-English resumes
   - Handles missing values

3. **Data Combining** (`src/data/combiner.py`)
   - Merges multiple datasets
   - Generates/reads candidate names and emails
   - Creates unified format

4. **Embedding Creation** (`src/embeddings/embedder.py`)
   - Converts resumes to vectors using `all-MiniLM-L6-v2`
   - 384-dimensional embeddings

5. **Vector Storage** (`src/embeddings/vectorstore.py`)
   - Stores in ChromaDB for fast similarity search
   - Persists to disk for reuse

6. **Semantic Search** (`src/ranker/cv_ranker.py`)
   - Finds top candidates using vector similarity
   - Fast keyword/skill matching

7. **LLM Evaluation** (`src/llm/groq_service.py`)
   - Deep analysis of each candidate's fit
   - Detailed evaluation with scores

8. **Hybrid Scoring** (`src/ranker/cv_ranker.py`)
   - Combines both scores (30% semantic + 70% LLM)
   - Weighted final score

9. **XAI Explanations** (`src/xai/explainer.py`)
   - Rule-based skill matching
   - SHAP feature importance
   - LIME text-level importance

10. **Ranking** (`src/ranker/cv_ranker.py`)
    - Returns top candidates sorted by final score
    - Includes metadata (name, email, category)

---

## 📊 API Endpoints

### `POST /rank`

Rank candidates against a job description.

**Request:**
```json
{
  "job_description": "Senior Data Engineer with Python, SQL, AWS; 5+ years",
  "top_k": 5,
  "include_explanations": true
}
```

**Response:**
```json
{
  "job_description": "...",
  "top_k": 5,
  "candidates": [
    {
      "final_score": 8.5,
      "llm_score": 9.0,
      "semantic_similarity": 0.85,
      "evaluation": "Score: 9/10...",
      "meta": {
        "id": "resume1_123",
        "name": "John Smith",
        "email": "john.smith456@gmail.com",
        "category": "INFORMATION-TECHNOLOGY",
        "source": "Resume.csv"
      },
      "preview": "...",
      "explanation": {
        "score_breakdown": {...},
        "skill_analysis": {...},
        "shap_analysis": {...},
        "lime_analysis": {...}
      }
    }
  ],
  "total_candidates_evaluated": 12
}
```

### `POST /explain`

Get detailed explanation for a specific candidate.

**Request:**
```json
{
  "candidate_id": "resume1_123",
  "job_description": "Senior Data Engineer..."
}
```

**Response:** Complete explanation with SHAP and LIME analysis.

### `GET /health`

Check service health and status.

**Response:**
```json
{
  "status": "healthy",
  "vectorstore_ready": true,
  "llm_ready": true,
  "total_resumes": 2646
}
```

### `POST /upload`

Upload PDF CVs (future feature).

---

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

### Embedding Settings
```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "resumes"
```

### LLM Settings
```python
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEMPERATURE = 0.1
GROQ_TOP_P = 0.9
GROQ_MAX_TOKENS = 1024
GROQ_SEED = 1  # For reproducibility
```

### Ranking Weights
```python
SEMANTIC_WEIGHT = 0.3  # 30% semantic similarity
LLM_WEIGHT = 0.7       # 70% LLM evaluation
DEFAULT_TOP_K = 5      # Default number of results
```

### Text Processing
```python
MIN_RESUME_LENGTH = 50
RESUME_HEAD_CHARS = 6000  # Characters from start for LLM
RESUME_TAIL_CHARS = 3000  # Characters from end for LLM
```

---

## 🎯 Features

### Core Features
- ✅ **Hybrid Ranking** - Combines semantic search + LLM analysis
- ✅ **Fast Semantic Search** - Vector similarity for quick filtering
- ✅ **Deep LLM Analysis** - Groq Cloud for detailed candidate evaluation
- ✅ **Candidate Names** - Automatic name/email generation or from CSV
- ✅ **Metadata Tracking** - Category, source, ID for each candidate

### XAI Features (Explainable AI)
- ✅ **Rule-Based Explanations** - Skill matching, score breakdowns
- ✅ **SHAP Analysis** - Feature-level importance (which skills matter)
- ✅ **LIME Analysis** - Word/phrase-level importance (which text matters)
- ✅ **Hybrid Approach** - Combines all three for comprehensive explanations

### API Features
- ✅ **REST API** - FastAPI with Swagger documentation
- ✅ **Health Checks** - Monitor service status
- ✅ **CORS Support** - Frontend integration ready
- ✅ **Error Handling** - Graceful error messages

### MLOps Features
- ✅ **MLflow Integration** - Experiment tracking ready
- ✅ **Metrics Calculation** - Ranking performance metrics
- ✅ **Reproducibility** - Seed-based LLM for consistent results

---

## 📊 Output Format

### Ranking Results

Each candidate includes:
- **final_score** (0-10) - Combined hybrid score
- **llm_score** (0-10) - LLM evaluation score
- **semantic_similarity** (0-1) - Vector similarity
- **evaluation** - Full LLM evaluation text
- **meta** - Candidate metadata:
  - `id` - Unique candidate ID
  - `name` - Candidate name
  - `email` - Candidate email
  - `category` - Resume category
  - `source` - Source CSV file
- **preview** - First 240 characters of resume
- **explanation** (optional) - XAI explanation with:
  - Score breakdown
  - Skill analysis
  - SHAP feature importance
  - LIME text importance
  - Strengths and weaknesses

---

## 🚀 Quick Start Examples

### Example 1: Basic Ranking (CLI)

```bash
python main.py
```

### Example 2: API Request

```bash
# Start server
python app.py

# In another terminal
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Data Engineer with Python, SQL, AWS",
    "top_k": 5
  }'
```

### Example 3: With Explanations

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Data Engineer",
    "top_k": 5,
    "include_explanations": true
  }'
```

---

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
- Set the environment variable (see Setup step 2)
- Or edit `src/utils/config.py` (not recommended for production)

### "File not found: data/Resume.csv"
- Make sure CSV files are in the `data/` folder
- Check file names match exactly

### "Vectorstore doesn't have names"
- The system will automatically recreate vectorstore with names
- Or manually delete `chroma_db/` folder and restart

### Slow performance
- First run creates embeddings (takes time)
- Subsequent runs use cached vectorstore
- LLM evaluation takes 2-5 seconds per candidate

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

---

## 📝 Notes

- **First run** will download the embedding model (~80MB)
- **Vectorstore** is saved in `chroma_db/` folder (reused on next run)
- **Logs** are saved in `logs/` folder
- **Candidate names** are deterministic (same ID = same name)
- **XAI features** require `shap` and `lime` packages (optional)

---

## 🔗 Key Technologies

- **FastAPI** - Modern Python web framework
- **LangChain** - LLM orchestration and chains
- **ChromaDB** - Vector database for embeddings
- **Groq Cloud** - Fast LLM inference
- **HuggingFace** - Embedding models
- **SHAP** - Feature importance explanations
- **LIME** - Text-level explanations
- **MLflow** - Experiment tracking

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when server is running)
- **Groq Console**: https://console.groq.com/
- **LangChain Docs**: https://python.langchain.com/
- **ChromaDB Docs**: https://www.trychroma.com/

---

## 🎉 Summary

This system provides:
- **Fast ranking** using hybrid semantic + LLM approach
- **Explainable results** with SHAP and LIME
- **Production-ready API** with FastAPI
- **Complete documentation** of all components

Perfect for HR teams, recruiters, or anyone needing intelligent CV ranking with explanations! 🚀
