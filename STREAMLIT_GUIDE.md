# Streamlit Frontend Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit requests
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

**Option A: Direct Service (Recommended - Faster)**
```bash
streamlit run streamlit_app.py
```

This uses the service directly (no API needed).

**Option B: Via API**
1. Start the FastAPI server first:
   ```bash
   python app.py
   # Or: uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. Then run Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. In the sidebar, make sure the API URL is correct (default: `http://localhost:8000`)

### 3. Access the App

Open your browser to: `http://localhost:8501`

## Features

### ✅ Professional Light Mode UI
- Clean, modern design
- Light color scheme
- Professional typography
- Smooth transitions

### ✅ Candidate Ranking
- Enter job description
- Select number of top candidates (1-20)
- View ranked results with scores

### ✅ Detailed Candidate Information
- Candidate name and email
- Final score, LLM score, semantic similarity
- Resume preview
- LLM evaluation text

### ✅ XAI Explanations (Optional)
- Score breakdown (semantic vs LLM contribution)
- Skill analysis (matched vs missing skills)
- Strengths and weaknesses
- SHAP analysis (feature importance)
- LIME analysis (word/phrase importance)

### ✅ Summary Metrics
- Average score
- Highest/Lowest scores
- Total candidates evaluated

## UI Components

### Main Page
- **Job Description Input**: Large text area for entering job requirements
- **Rank Button**: Primary action button to start ranking
- **Results Section**: Displays ranked candidates with expandable details

### Sidebar
- **Configuration**: Connection method (Direct/API)
- **Ranking Settings**: Top K candidates, XAI explanations toggle
- **System Info**: Total resumes in database

### Candidate Cards
- **Header**: Name, email, category
- **Score Badge**: Color-coded (High/Medium/Low match)
- **Metrics**: Final score, LLM score, semantic similarity
- **Expandable Sections**: Preview, evaluation, XAI explanations

## Color Scheme

- **High Match** (≥7.5): Green badge
- **Medium Match** (5.0-7.5): Yellow badge
- **Low Match** (<5.0): Red badge

## Troubleshooting

### Service Not Ready
- Make sure the vectorstore is initialized
- Check that data files are in `data/` directory
- Verify GROQ_API_KEY is set in environment

### API Connection Issues
- Ensure FastAPI server is running
- Check API URL in sidebar
- Verify CORS settings in `app.py`

### Slow Performance
- Use direct service mode (faster than API)
- Reduce number of candidates requested
- Disable XAI explanations for faster results

## Customization

### Change Colors
Edit the CSS in `streamlit_app.py` (lines 20-120):
```python
.stButton > button {
    background-color: #3b82f6;  # Change this
    ...
}
```

### Add More Metrics
Edit the `display_candidate()` function to show additional information.

### Modify Layout
Adjust column widths and spacing in the Streamlit layout functions.

