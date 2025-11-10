# 🧠 TALENTGRAPH — Model Development Branch

This branch contains the core **machine learning model development** for the TALENTGRAPH platform.  
The goal is to train and evaluate AI models that can **analyze resumes, extract skills, and rank candidates** based on their fit for specific job roles.

---

## 🎯 Objective

To develop and fine-tune the **CV Ranking Model**, the central engine behind TALENTGRAPH’s candidate-job matching pipeline.

The model:
- Uses NLP to understand resumes and job descriptions
- Builds vector embeddings using pretrained transformer models
- Computes semantic similarity between candidates and roles
- Produces ranked candidate lists per job

---

## 🧩 Notebook: `cv_ranker_notebook.ipynb`

This Jupyter notebook contains:
- Data loading and preprocessing steps  
- Text cleaning and normalization for resumes/jobs  
- Embedding generation using transformer-based models (e.g. `Sentence-BERT`)  
- Similarity computation and ranking logic  
- Evaluation metrics and model performance visualization  

---

## 🧠 Techniques & Libraries

- **Transformers / Sentence-BERT** for semantic embeddings  
- **scikit-learn** for evaluation and vector operations  
- **pandas / numpy** for data handling  
- **tqdm / matplotlib** for progress and visualization  

---

## ⚙️ Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
