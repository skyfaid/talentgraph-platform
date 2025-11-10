"""
PDF parsing utilities for extracting text from CV PDFs.
"""
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as string
        
    Raises:
        ImportError: If required PDF libraries are not installed
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try different PDF parsing libraries in order of preference
    try:
        # Try pdfplumber first (better for tables and complex layouts)
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except ImportError:
        pass
    
    try:
        # Fall back to PyPDF2
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        pass
    
    try:
        # Fall back to pypdf
        import pypdf
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        raise ImportError(
            "No PDF parsing library found. Please install one of: "
            "pdfplumber, PyPDF2, or pypdf"
        )


def extract_text_from_multiple_pdfs(pdf_paths: List[Path]) -> List[dict]:
    """
    Extract text from multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        List of dictionaries with 'path', 'text', and 'filename' keys
    """
    results = []
    
    for pdf_path in pdf_paths:
        try:
            text = extract_text_from_pdf(pdf_path)
            results.append({
                'path': str(pdf_path),
                'filename': pdf_path.name,
                'text': text
            })
        except Exception as e:
            print(f"⚠️ Error processing {pdf_path.name}: {e}")
            results.append({
                'path': str(pdf_path),
                'filename': pdf_path.name,
                'text': '',
                'error': str(e)
            })
    
    return results

