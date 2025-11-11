"""
Streamlit Frontend for CV Ranking System

Professional, light-mode UI for ranking CVs against job descriptions.
"""
import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for direct service access
sys.path.insert(0, str(Path(__file__).parent))

# Setup logger
import logging
logger = logging.getLogger(__name__)

# Try to use service directly (faster) or fall back to API
API_URL = "http://localhost:8000"  # Default API URL
try:
    from src.api import get_service
    USE_DIRECT_SERVICE = True
except ImportError:
    USE_DIRECT_SERVICE = False

# Page configuration
st.set_page_config(
    page_title="CV Ranking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional light mode
st.markdown("""
    <style>
    /* Main theme - Light mode */
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }
    
    /* All text should be dark */
    body, p, div, span, label {
        color: #1f2937 !important;
    }
    
    /* Headers */
    h1 {
        color: #1f2937 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #374151 !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h3 {
        color: #4b5563 !important;
        font-weight: 600;
    }
    
    /* Streamlit text elements */
    .stMarkdown, .stText, .stMarkdown p, .stMarkdown div {
        color: #1f2937 !important;
    }
    
    /* Sidebar text - keep white */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    .css-1d391kg, 
    .css-1d391kg p, 
    .css-1d391kg div, 
    .css-1d391kg span,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar background - dark */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important;
    }
    
    .css-1d391kg {
        background-color: #1f2937 !important;
    }
    
    /* Input labels - dark for main, white for sidebar */
    label, .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label {
        color: #374151 !important;
        font-weight: 500;
    }
    
    /* Sidebar input labels - white */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
    }
    
    /* Sidebar inputs - white text */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stSelectbox select {
        color: #ffffff !important;
        background-color: #374151 !important;
    }
    
    /* Sidebar metrics - white */
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric div {
        color: #ffffff !important;
    }
    
    /* Sidebar success/info/error - white text */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stError {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSuccess p,
    [data-testid="stSidebar"] .stInfo p,
    [data-testid="stSidebar"] .stWarning p,
    [data-testid="stSidebar"] .stError p {
        color: #ffffff !important;
    }
    
    /* Metric labels and values */
    .stMetric label, .stMetric div {
        color: #1f2937 !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        color: #1f2937 !important;
    }
    
    /* Code blocks */
    .stCodeBlock, code {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
    }
    
    /* Sidebar - dark background with white text */
    .css-1d391kg {
        background-color: #1f2937 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Text inputs - dark background with white text */
    .stTextArea > div > div > textarea {
        border: 1px solid #4b5563;
        border-radius: 6px;
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #4b5563;
        border-radius: 6px;
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }
    
    /* Placeholder text - lighter gray */
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        border: 1px solid #d1d5db;
        border-radius: 6px;
    }
    
    /* Cards/Metrics */
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Candidate cards */
    .candidate-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.3s;
        color: #1f2937 !important;
    }
    
    .candidate-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .candidate-card p, .candidate-card div, .candidate-card span {
        color: #1f2937 !important;
    }
    
    /* All Streamlit widgets text - dark for main area */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        color: #1f2937 !important;
    }
    
    /* Textarea in main area - make text white if dark background */
    .stTextArea textarea {
        color: #ffffff !important;
        background-color: #1f2937 !important;
    }
    
    /* Text input in main area */
    .stTextInput input {
        color: #ffffff !important;
        background-color: #1f2937 !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #1f2937 !important;
    }
    
    /* Warning, info, success, error text */
    .stAlert, .stAlert p, .stAlert div {
        color: #1f2937 !important;
    }
    
    /* Table text */
    .stDataFrame, .stTable {
        color: #1f2937 !important;
    }
    
    /* Divider */
    hr {
        border-color: #e5e7eb;
    }
    
    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .score-high {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .score-medium {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .score-low {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
    }
    
    /* Error messages */
    .stError {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def get_service_instance():
    """Get service instance (direct or via API)."""
    if USE_DIRECT_SERVICE:
        if 'service' not in st.session_state:
            st.session_state.service = get_service()
            if not st.session_state.service.is_ready():
                st.session_state.service.initialize()
        return st.session_state.service
    return None


def rank_via_api(job_description: str, top_k: int, include_explanations: bool, api_url: str = None) -> Dict:
    """Rank candidates via FastAPI endpoint."""
    if api_url is None:
        api_url = API_URL
    try:
        response = requests.post(
            f"{api_url}/rank",
            json={
                "job_description": job_description,
                "top_k": top_k,
                "include_explanations": include_explanations
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None


def rank_via_service(job_description: str, top_k: int, include_explanations: bool) -> Dict:
    """Rank candidates via direct service."""
    try:
        service = get_service_instance()
        if not service or not service.is_ready():
            st.error("Service not ready. Please check initialization.")
            return None
        
        # Rank candidates
        results = service.ranker.rank_resumes(
            job_description=job_description,
            top_k=top_k
        )
        
        # Generate explanations if requested
        explanations = {}
        if include_explanations:
            from src.xai import RankingExplainer
            explainer = RankingExplainer()
            
            for r in results:
                candidate_id = r['meta'].get('id')
                try:
                    # Get full resume text
                    doc_results = service.vectorstore.similarity_search_with_relevance_scores(
                        r['preview'], k=1
                    )
                    resume_text = doc_results[0][0].page_content if doc_results else r['preview']
                    
                    explanation = explainer.generate_explanation(
                        candidate_result=r,
                        job_description=job_description,
                        resume_text=resume_text,
                        vectorstore=service.vectorstore,
                        ranker=service.ranker
                    )
                    explanations[candidate_id] = explanation
                except Exception as e:
                    st.warning(f"Could not generate explanation for {candidate_id}: {e}")
        
        # Format response
        candidates = []
        for r in results:
            candidate = {
                "final_score": r['final_score'],
                "llm_score": r['llm_score'],
                "semantic_similarity": r['semantic_similarity'],
                "evaluation": r['evaluation'],
                "meta": r['meta'],
                "preview": r['preview']
            }
            if r['meta'].get('id') in explanations:
                candidate['explanation'] = explanations[r['meta'].get('id')]
            candidates.append(candidate)
        
        return {
            "job_description": job_description,
            "top_k": top_k,
            "candidates": candidates,
            "total_candidates_evaluated": len(results)
        }
    except Exception as e:
        st.error(f"Error ranking candidates: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def format_score(score: float) -> tuple:
    """Format score with color coding."""
    if score >= 7.5:
        return "score-high", "High Match"
    elif score >= 5.0:
        return "score-medium", "Medium Match"
    else:
        return "score-low", "Low Match"


def start_interview_flow(candidate: Dict, job_description: str, index: int = 1):
    """Start interview flow for a candidate."""
    meta = candidate.get('meta', {})
    candidate_id = meta.get('id', 'unknown')
    candidate_name = meta.get('name', 'Unknown Candidate')
    candidate_email = meta.get('email', 'N/A')
    candidate_resume = candidate.get('preview', '') + "\n\n" + candidate.get('evaluation', '')
    
    # Use unique key with index to avoid conflicts
    unique_key = f"{candidate_id}_{index}"
    
    # Show interview setup section prominently
    st.markdown("---")
    st.markdown("## 🎤 Interview Setup")
    st.success(f"✅ Interview setup for: **{candidate_name}** ({candidate_email})")
    st.info("Select the number of questions below and click 'Start Interview Session' to begin.")
    
    # Question count selection (5, 7, or 10 questions)
    question_count = st.selectbox(
        "Number of Questions",
        [5, 7, 10],
        index=0,  # Default to 5
        key=f"question_count_{unique_key}",
        help="5 questions: 3 behavioral + 2 technical | 7 questions: 4 behavioral + 3 technical | 10 questions: 6 behavioral + 4 technical"
    )
    
    # Start interview button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_btn_clicked = st.button("🚀 Start Interview Session", key=f"confirm_interview_{unique_key}", type="primary", use_container_width=True)
    
    if start_btn_clicked:
        with st.spinner("🔄 Creating interview session and generating questions..."):
            try:
                # Get API URL from session state or use default
                current_api_url = st.session_state.get('api_url', API_URL)
                # Call interview API
                response = requests.post(
                    f"{current_api_url}/interview/start",
                    json={
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "candidate_email": candidate_email,
                        "job_description": job_description,
                        "candidate_resume": candidate_resume,
                        "interview_type": "unified",  # Use unified mode
                        "question_count": question_count  # Pass question count
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                session_id = result.get("session_id")
                st.session_state[f"interview_session_{unique_key}"] = session_id
                st.session_state[f"interview_active_{unique_key}"] = True
                st.success(f"✅ Interview session created! {result.get('message', '')}")
                # Clear the start flag
                st.session_state[f"start_interview_{candidate_id}_{index}"] = False
                st.rerun()
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to start interview: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        st.error(f"Error details: {error_detail}")
                    except:
                        st.error(f"Status code: {e.response.status_code}")
    
    # If interview is active, show interview interface
    if st.session_state.get(f"interview_active_{unique_key}", False):
        session_id = st.session_state.get(f"interview_session_{unique_key}")
        if session_id:
            st.markdown("---")
            show_interview_interface(session_id, unique_key)
        else:
            st.warning("Interview session not found. Please start a new interview.")
    
    # Add button to close interview setup (if not started yet)
    if not st.session_state.get(f"interview_active_{unique_key}", False):
        if st.button("❌ Cancel Interview", key=f"cancel_interview_{unique_key}"):
            # Clear interview state
            if st.session_state.get("active_interview_key") == unique_key:
                st.session_state["active_interview_key"] = None
                st.session_state["active_interview_candidate"] = None
                st.session_state["active_interview_index"] = None
                st.session_state["active_interview_job_desc"] = None
            st.rerun()


def show_interview_interface(session_id: str, candidate_id: str):
    """Show the interview interface for answering questions."""
    try:
        # Get API URL from session state or use default
        current_api_url = st.session_state.get('api_url', API_URL)
        # Get interview status
        status_response = requests.get(f"{current_api_url}/interview/{session_id}/status", timeout=30)
        status_response.raise_for_status()
        status = status_response.json()
        
        st.markdown("---")
        st.subheader("📝 Interview in Progress")
        
        # Show progress
        total_q = status.get("total_questions", 0)
        answered_q = status.get("answered_questions", 0)
        progress = answered_q / total_q if total_q > 0 else 0
        
        st.progress(progress)
        st.caption(f"Progress: {answered_q}/{total_q} questions answered")
        
        # Show scores if available
        if status.get("overall_score") is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{status.get('overall_score', 0):.2f}/10")
            with col2:
                st.metric("Technical Score", f"{status.get('technical_score', 0):.2f}/10")
            with col3:
                st.metric("Behavioral Score", f"{status.get('behavioral_score', 0):.2f}/10")
        
        # Get next question
        current_q_id = status.get("current_question_id")
        if current_q_id is not None:
            try:
                question_response = requests.get(
                    f"{current_api_url}/interview/{session_id}/next-question",
                    timeout=30
                )
                question_response.raise_for_status()
                question = question_response.json()
                
                st.markdown("### Current Question")
                q_type = question.get("question_type", "technical")
                type_badge = "🔧 Technical" if q_type == "technical" else "💬 Behavioral" if q_type == "behavioral" else "❓ Follow-up"
                st.markdown(f"**{type_badge} Question #{current_q_id + 1}**")
                
                question_text = question.get('question', 'N/A')
                st.markdown(f"**{question_text}**")
                
                # Voice reading (TTS)
                try:
                    from src.voice import get_tts_service
                    tts_service = get_tts_service()
                    
                    # Check if we already generated audio for this question
                    audio_key = f"audio_{session_id}_{current_q_id}"
                    error_key = f"audio_error_{session_id}_{current_q_id}"
                    
                    # Generate audio section
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown("### 🔊 Listen to Question")
                        
                        if audio_key not in st.session_state:
                            # Generate audio on first display
                            with st.spinner("🎤 Generating voice..."):
                                audio_bytes = tts_service.text_to_speech(
                                    text=question_text,
                                    provider="auto"  # Tries gTTS first (simpler), then edge-tts
                                )
                                if audio_bytes and len(audio_bytes) > 0:
                                    st.session_state[audio_key] = audio_bytes
                                    # Clear any previous error
                                    if error_key in st.session_state:
                                        del st.session_state[error_key]
                                    st.success("✅ Audio ready!")
                                else:
                                    st.session_state[error_key] = "Failed to generate audio. Please check your internet connection."
                        
                        # Display audio player if available
                        if audio_key in st.session_state:
                            audio_bytes = st.session_state[audio_key]
                            if audio_bytes and len(audio_bytes) > 0:
                                # Large, prominent audio player
                                st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                                st.caption("🎧 **Click the play button above to hear the question read aloud**")
                                
                                # Also add a play button for easier access
                                if st.button("▶️ Play Question Audio", key=f"play_{session_id}_{current_q_id}", type="primary"):
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                                    st.rerun()
                            else:
                                st.warning("⚠️ Audio generation failed. Please check your internet connection.")
                        elif error_key in st.session_state:
                            st.warning(f"⚠️ {st.session_state[error_key]}")
                            if st.button("🔄 Retry Audio Generation", key=f"retry_audio_{session_id}_{current_q_id}"):
                                if error_key in st.session_state:
                                    del st.session_state[error_key]
                                st.rerun()
                except Exception as e:
                    # TTS not available, show error
                    st.warning(f"⚠️ Voice reading unavailable: {str(e)}")
                    st.info("💡 You can still read the question above and answer it.")
                
                # Show hints if available
                if question.get("expected_points"):
                    with st.expander("💡 Expected Answer Points"):
                        st.write(question.get("expected_points"))
                
                if question.get("star_required"):
                    with st.expander("⭐ STAR Format Guide"):
                        st.write(question.get("star_required"))
                
                # Answer input - Text or Voice
                answer_key = f"answer_{session_id}_{current_q_id}"
                clear_answer_key = f"clear_answer_{session_id}_{current_q_id}"
                
                # Check if we need to clear the answer (from previous submit)
                if st.session_state.get(clear_answer_key, False):
                    # Delete the answer key and clear the flag before creating widget
                    if answer_key in st.session_state:
                        del st.session_state[answer_key]
                    st.session_state[clear_answer_key] = False
                
                # Initialize answer from session state if available
                # This ensures voice transcription is preserved
                if answer_key not in st.session_state:
                    st.session_state[answer_key] = ""
                
                answer = st.session_state.get(answer_key, "")
                
                # Tabs for text vs voice input
                input_tab1, input_tab2 = st.tabs(["✍️ Type Answer", "🎤 Voice Answer"])
                
                with input_tab1:
                    # Text area - this will sync with session state via key
                    answer = st.text_area(
                        "Your Answer:",
                        value=st.session_state[answer_key],
                        height=200,
                        key=answer_key,
                        placeholder="Type your answer here..."
                    )
                    # Update session state when user types
                    if answer != st.session_state.get(answer_key, ""):
                        st.session_state[answer_key] = answer
                
                with input_tab2:
                    st.info("🎤 Record your answer using your microphone")
                    st.caption("💡 Click the microphone button below to start recording")
                    
                    try:
                        from audio_recorder_streamlit import audio_recorder
                        
                        # Audio recorder with better error handling
                        try:
                            audio_data = audio_recorder(
                                text="🎤 Click to record",
                                recording_color="#e74c3c",
                                neutral_color="#6c757d",
                                icon_name="microphone",
                                icon_size="2x",
                                pause_threshold=3.0
                            )
                        except Exception as recorder_error:
                            st.error(f"❌ Microphone error: {recorder_error}")
                            st.info("💡 Please check browser permissions. Click the lock icon in address bar and allow microphone access.")
                            audio_data = None
                        
                        # Show status
                        if audio_data is not None:
                            if len(audio_data) > 0:
                                st.success(f"✅ Audio recorded ({len(audio_data)} bytes)")
                                
                                # Save audio to temporary file
                                import tempfile
                                import os
                                from pydub import AudioSegment
                                import io
                                
                                # Save audio to temp file
                                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                                temp_audio.write(audio_data)
                                temp_audio.close()
                                
                                # Convert to WAV format if needed (Whisper expects WAV)
                                try:
                                    # Try to load and convert to WAV
                                    audio = AudioSegment.from_file(io.BytesIO(audio_data))
                                    # Ensure it's mono and 16kHz (Whisper's preferred format)
                                    audio = audio.set_channels(1).set_frame_rate(16000)
                                    
                                    # Save as WAV
                                    wav_path = temp_audio.name.replace(".wav", "_converted.wav")
                                    audio.export(wav_path, format="wav")
                                    
                                    # Use converted file
                                    audio_file_path = wav_path
                                except Exception as conv_error:
                                    # If conversion fails, try original file
                                    logging.warning(f"Audio conversion failed: {conv_error}, using original")
                                    audio_file_path = temp_audio.name
                                
                                # Convert speech to text
                                try:
                                    from src.voice import get_stt_service
                                    stt_service = get_stt_service()
                                    
                                    if not stt_service.whisper_available or not stt_service.model:
                                        st.error("⚠️ Whisper not available. Please install: pip install openai-whisper")
                                    else:
                                        with st.spinner("🔄 Converting speech to text..."):
                                            transcription = stt_service.speech_to_text(audio_file_path)
                                            
                                            if transcription.get("text"):
                                                transcribed_text = transcription['text'].strip()
                                                # Update answer text area with transcribed text
                                                st.success(f"✅ Transcribed: {transcribed_text[:100]}...")
                                                
                                                # CRITICAL: Set the answer in session state
                                                st.session_state[answer_key] = transcribed_text
                                                
                                                # Also show in the text area tab
                                                st.info(f"📝 Your transcribed answer: {transcribed_text[:200]}...")
                                                
                                                # Force update the answer variable
                                                answer = transcribed_text
                                                
                                                # Show success and refresh
                                                st.success("✅ Answer saved! You can edit it in the 'Type Answer' tab or submit it now.")
                                                
                                                # Small delay then refresh to show transcribed text in text area
                                                import time
                                                time.sleep(0.5)
                                                st.rerun()
                                            else:
                                                error_msg = transcription.get("error", "Unknown error")
                                                st.error(f"❌ Could not transcribe audio: {error_msg}")
                                                st.info("💡 Try speaking more clearly or check your microphone.")
                                except Exception as e:
                                    st.error(f"❌ Speech-to-text error: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                    st.info("💡 Please type your answer instead or try recording again.")
                                
                                # Clean up temp files
                                try:
                                    os.unlink(temp_audio.name)
                                    if audio_file_path != temp_audio.name:
                                        os.unlink(audio_file_path)
                                except:
                                    pass
                            else:
                                st.warning("⚠️ No audio data received. Please try recording again.")
                    except ImportError:
                        st.error("❌ Voice input not available")
                        st.info("💡 Please install: `pip install audio-recorder-streamlit`")
                        answer = st.text_area(
                            "Your Answer (Voice input unavailable):",
                            height=200,
                            key=f"{answer_key}_voice",
                            placeholder="Voice input not available. Please use the 'Type Answer' tab."
                        )
                    except Exception as e:
                        st.error(f"❌ Voice input error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        answer = st.session_state.get(answer_key, "")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("✅ Submit Answer", key=f"submit_{session_id}_{current_q_id}", type="primary"):
                        # Get answer from session state (most reliable - works for both typed and voice)
                        final_answer = st.session_state.get(answer_key, "").strip()
                        if not final_answer:
                            # Fallback to answer variable
                            final_answer = answer.strip() if answer else ""
                        
                        if final_answer:
                            with st.spinner("🔄 Evaluating your answer..."):
                                try:
                                    submit_response = requests.post(
                                        f"{current_api_url}/interview/submit-answer",
                                        json={
                                            "session_id": session_id,
                                            "question_id": current_q_id,
                                            "answer": final_answer  # Use final_answer from session state
                                        },
                                        timeout=60
                                    )
                                    submit_response.raise_for_status()
                                    evaluation = submit_response.json()
                                    
                                    # Show evaluation
                                    st.markdown("### 📊 Answer Evaluation")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Score", f"{evaluation.get('score', 0):.2f}/10")
                                    with col2:
                                        rec = evaluation.get('recommendation', 'needs_followup')
                                        rec_emoji = "✅" if rec == "strong" else "⚠️" if rec == "weak" else "❓"
                                        st.metric("Recommendation", f"{rec_emoji} {rec.title()}")
                                    
                                    # Strengths and weaknesses
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.success("**✅ Strengths:**")
                                        for strength in evaluation.get('strengths', [])[:3]:
                                            st.write(f"• {strength}")
                                    with col2:
                                        st.warning("**⚠️ Areas to Improve:**")
                                        for weakness in evaluation.get('weaknesses', [])[:3]:
                                            st.write(f"• {weakness}")
                                    
                                    # Feedback
                                    st.info(f"**💬 Feedback:**\n\n{evaluation.get('feedback', '')}")
                                    
                                    # Mark answer for clearing on next rerun (before widget creation)
                                    st.session_state[clear_answer_key] = True
                                    st.rerun()
                                    
                                except requests.exceptions.RequestException as e:
                                    st.error(f"❌ Failed to submit answer: {e}")
                        else:
                            st.warning("Please enter an answer before submitting.")
                
                with col2:
                    if st.button("⏭️ Skip Question", key=f"skip_{session_id}_{current_q_id}"):
                        st.info("Skipping questions is not recommended. Please provide an answer.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to get question: {e}")
        else:
            # Interview completed
            st.success("🎉 Interview Completed!")
            
            # Get final report
            if st.button("📊 View Full Report", key=f"report_{session_id}"):
                try:
                    report_response = requests.get(
                        f"{current_api_url}/interview/{session_id}/report",
                        timeout=30
                    )
                    report_response.raise_for_status()
                    report = report_response.json()
                    
                    st.markdown("### 📋 Interview Report")
                    
                    # Scores
                    scores = report.get("scores", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{scores.get('overall_score', 0):.2f}/10")
                    with col2:
                        st.metric("Technical Score", f"{scores.get('technical_score', 0):.2f}/10")
                    with col3:
                        st.metric("Behavioral Score", f"{scores.get('behavioral_score', 0):.2f}/10")
                    
                    # All Q&A
                    st.markdown("### All Questions & Answers")
                    questions = report.get("questions", [])
                    answers = report.get("answers", [])
                    
                    for i, question in enumerate(questions):
                        with st.expander(f"Question {i+1}: {question.get('question', '')[:50]}..."):
                            st.write(f"**Type:** {question.get('type', 'N/A')}")
                            st.write(f"**Question:** {question.get('question', 'N/A')}")
                            
                            # Find corresponding answer
                            answer_data = next((a for a in answers if a.get('question_id') == i), None)
                            if answer_data:
                                st.write(f"**Answer:** {answer_data.get('answer', 'N/A')}")
                                eval_data = answer_data.get('evaluation', {})
                                if eval_data:
                                    st.write(f"**Score:** {eval_data.get('score', 0):.2f}/10")
                                    st.write(f"**Feedback:** {eval_data.get('feedback', 'N/A')}")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to get report: {e}")
            
            # Reset interview state
            if st.button("🔄 Start New Interview", key=f"reset_{session_id}"):
                # Find and clear all interview states for this session
                for key in list(st.session_state.keys()):
                    if key.startswith(f"interview_active_") or key.startswith(f"interview_session_"):
                        if st.session_state.get(key) == session_id or key.endswith(f"_{candidate_id}"):
                            del st.session_state[key]
                st.rerun()
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to get interview status: {e}")


def display_candidate(candidate: Dict, index: int, job_description: str = ""):
    """Display a candidate card with interview button."""
    final_score = candidate['final_score']
    llm_score = candidate['llm_score']
    semantic = candidate['semantic_similarity']
    meta = candidate.get('meta', {})
    name = meta.get('name', 'Unknown Candidate')
    email = meta.get('email', 'N/A')
    category = meta.get('category', 'Unknown')
    candidate_id = meta.get('id', 'unknown')
    
    score_class, score_label = format_score(final_score)
    
    # Candidate card with interview button
    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
    
    with col_header1:
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="margin: 0; color: #1f2937 !important;">#{index}. {name}</h3>
            <p style="margin: 0.25rem 0; color: #4b5563 !important; font-size: 0.875rem;">{email}</p>
            <p style="margin: 0.25rem 0; color: #4b5563 !important; font-size: 0.875rem;">Category: {category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown(f"""
        <div style="text-align: center; margin-top: 0.5rem;">
            <div class="score-badge {score_class}" style="font-size: 1.25rem; padding: 0.5rem 1rem; color: inherit !important; display: inline-block;">
                {final_score:.2f}/10
            </div>
            <p style="margin: 0.25rem 0; color: #4b5563 !important; font-size: 0.75rem;">{score_label}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        # Store candidate info in session state for interview
        interview_key = f"interview_{candidate_id}_{index}"
        if interview_key not in st.session_state:
            st.session_state[interview_key] = {
                "candidate_id": candidate_id,
                "candidate_name": name,
                "candidate_email": email,
                "job_description": job_description,
                "candidate_resume": candidate.get('preview', '') + "\n\n" + candidate.get('evaluation', '')
            }
        
        # Interview button - use index to ensure unique key
        button_key = f"interview_btn_{candidate_id}_{index}"
        button_clicked = st.button("🎤 Start Interview", key=button_key, use_container_width=True)
        
        if button_clicked:
            # Store candidate data and set interview state immediately
            unique_key = f"{candidate_id}_{index}"
            flag_key = f"start_interview_{candidate_id}_{index}"
            
            # Store candidate data in session state
            st.session_state[f"candidate_data_{unique_key}"] = candidate
            st.session_state["active_interview_key"] = unique_key
            st.session_state["active_interview_candidate"] = candidate
            st.session_state["active_interview_index"] = index
            st.session_state["active_interview_job_desc"] = job_description
            
            # Set flag (for backward compatibility)
            st.session_state[flag_key] = True
            st.session_state["current_interview_candidate"] = unique_key
            
            # Force rerun to show interview flow
            st.rerun()
    
    # Score breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Score", f"{final_score:.2f}", f"{score_label}")
    with col2:
        st.metric("LLM Score", f"{llm_score:.2f}/10")
    with col3:
        st.metric("Semantic Similarity", f"{semantic:.3f}")
    
    # Preview
    with st.expander("📄 Resume Preview", expanded=False):
        st.text(candidate.get('preview', 'N/A')[:500] + "...")
    
    # LLM Evaluation
    with st.expander("🤖 LLM Evaluation", expanded=False):
        st.markdown(candidate.get('evaluation', 'N/A'))
    
    # XAI Explanation
    if 'explanation' in candidate:
        explanation = candidate['explanation']
        with st.expander("🔍 Detailed Explanation (XAI)", expanded=False):
            # Score breakdown
            st.subheader("Score Breakdown")
            score_breakdown = explanation.get('score_breakdown', {})
            components = score_breakdown.get('components', {})
            semantic_comp = components.get('semantic', {})
            llm_comp = components.get('llm', {})
            col1, col2 = st.columns(2)
            with col1:
                semantic_contrib = semantic_comp.get('contribution', 0)
                st.metric("Semantic Contribution", f"{semantic_contrib:.2f}")
            with col2:
                llm_contrib = llm_comp.get('contribution', 0)
                st.metric("LLM Contribution", f"{llm_contrib:.2f}")
            
            # Skill analysis
            st.subheader("Skill Analysis")
            skill_analysis = explanation.get('skill_analysis', {})
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Matched Skills:**")
                matched = skill_analysis.get('matched_skills', [])
                if matched:
                    st.write(", ".join(matched))
                else:
                    st.write("None")
            with col2:
                st.write("**Missing Skills:**")
                missing = skill_analysis.get('missing_skills', [])
                if missing:
                    st.write(", ".join(missing))
                else:
                    st.write("None")
            
            # Strengths & Weaknesses
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Strengths")
                insights = explanation.get('insights', {})
                strengths = insights.get('strengths', [])
                if strengths:
                    for strength in strengths[:5]:
                        st.write(f"• {strength}")
                else:
                    st.write("None identified")
            with col2:
                st.subheader("⚠️ Weaknesses")
                insights = explanation.get('insights', {})
                weaknesses = insights.get('weaknesses', [])
                if weaknesses:
                    for weakness in weaknesses[:5]:
                        st.write(f"• {weakness}")
                else:
                    st.write("None identified")
            
            # SHAP Analysis
            if explanation.get('shap_analysis', {}).get('available'):
                st.subheader("📊 SHAP Analysis")
                shap_data = explanation['shap_analysis'].get('skill_importance', {})
                if shap_data:
                    top_positive = shap_data.get('top_positive', [])
                    if top_positive:
                        st.write("**Top Contributing Skills:**")
                        st.write(", ".join(top_positive[:5]))
            
            # LIME Analysis
            if explanation.get('lime_analysis', {}).get('available'):
                st.subheader("🔬 LIME Analysis")
                lime_data = explanation['lime_analysis'].get('text_importance', {})
                if lime_data:
                    top_words = lime_data.get('top_contributing_words', [])
                    if top_words:
                        st.write("**Important Words/Phrases:**")
                        st.write(", ".join(top_words[:8]))
    
    st.divider()


def main():
    """Main Streamlit app."""
    # Header
    st.title("🎯 CV Ranking System")
    st.markdown("**AI-Powered Candidate Ranking with Explainable AI**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection method
        if USE_DIRECT_SERVICE:
            st.success("✅ Using Direct Service")
        else:
            st.info("🌐 Using API Endpoint")
            # Use session state to store API URL
            if 'api_url' not in st.session_state:
                st.session_state.api_url = API_URL
            api_url_input = st.text_input("API URL", value=st.session_state.api_url)
            if api_url_input:
                st.session_state.api_url = api_url_input
        
        st.divider()
        
        # Settings
        st.subheader("Ranking Settings")
        top_k = st.slider("Number of Top Candidates", 1, 20, 5)
        include_explanations = st.checkbox("Include XAI Explanations", value=False)
        
        st.divider()
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        This system uses:
        - **Semantic Search** (30%)
        - **LLM Evaluation** (70%)
        - **XAI Explanations** (SHAP + LIME)
        """)
        
        if USE_DIRECT_SERVICE:
            service = get_service_instance()
            if service:
                total_resumes = service.get_total_resumes()
                if total_resumes:
                    st.metric("Total Resumes", total_resumes)
    
    # Main content
    # Job description input
    st.subheader("📝 Job Description")
    # Use stored job description if available, otherwise use empty string
    default_job_desc = st.session_state.get("last_job_description", "")
    job_description = st.text_area(
        "Enter the job description to rank candidates against:",
        value=default_job_desc,
        height=150,
        placeholder="Example: Senior Data Engineer with Python, SQL, AWS; 5+ years experience; leadership skills a plus.",
        help="Provide a detailed job description including required skills, experience level, and any other relevant requirements."
    )
    
    # Check for interview button clicks FIRST (works even after ranking)
    # This ensures interview flow appears when button is clicked
    candidates_list = []
    current_job_desc = job_description
    
    # Check if any interview button was clicked (from previously ranked candidates)
    for key in list(st.session_state.keys()):
        if key.startswith("start_interview_") and st.session_state.get(key, False):
            # Extract candidate_id and index from key
            parts = key.replace("start_interview_", "").rsplit("_", 1)
            if len(parts) == 2:
                candidate_id_part = parts[0]
                idx_part = parts[1]
                try:
                    idx = int(idx_part)
                    # Set persistent interview state
                    unique_key = f"{candidate_id_part}_{idx}"
                    # Try to get candidate from stored results or session state
                    stored_candidate = st.session_state.get(f"candidate_data_{unique_key}")
                    if stored_candidate:
                        st.session_state["active_interview_key"] = unique_key
                        st.session_state["active_interview_candidate"] = stored_candidate
                        st.session_state["active_interview_index"] = idx
                        st.session_state["active_interview_job_desc"] = st.session_state.get("last_job_description", job_description)
                        st.session_state[key] = False
                        st.rerun()
                except:
                    pass
    
    # Show active interview if it exists (BEFORE ranking section)
    active_interview_key = st.session_state.get("active_interview_key")
    if active_interview_key:
        active_candidate = st.session_state.get("active_interview_candidate")
        active_idx = st.session_state.get("active_interview_index", 1)
        active_job = st.session_state.get("active_interview_job_desc", job_description)
        
        if active_candidate:
            st.markdown("---")
            st.markdown("## 🎤 Interview Setup")
            start_interview_flow(active_candidate, active_job, active_idx)
            st.divider()
    
    # Rank button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        rank_button = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)
    
    # Check if we have stored candidates to display (persists across reruns)
    stored_candidates = st.session_state.get("ranked_candidates", [])
    stored_job_desc = st.session_state.get("last_job_description", "")
    
    # Results section
    if rank_button:
        if not job_description.strip():
            st.warning("⚠️ Please enter a job description before ranking.")
        else:
            with st.spinner("🔄 Ranking candidates... This may take a moment."):
                # Rank candidates
                if USE_DIRECT_SERVICE:
                    results = rank_via_service(job_description, top_k, include_explanations)
                else:
                    # Get current API URL from session state or use default
                    current_api_url = st.session_state.get('api_url', API_URL)
                    results = rank_via_api(job_description, top_k, include_explanations, current_api_url)
                
                if results:
                    num_shown = len(results['candidates'])
                    num_requested = results.get('top_k', top_k)
                    
                    if num_shown < num_requested:
                        st.warning(
                            f"⚠️ Requested {num_requested} candidates, but only {num_shown} met the minimum quality threshold (4.0/10). "
                            f"Low-scoring candidates were filtered out to avoid showing poor matches."
                        )
                    else:
                        st.success(f"✅ Found {results['total_candidates_evaluated']} candidates, showing top {num_shown}")
                    
                    # Store results in session state for persistence
                    st.session_state["ranked_candidates"] = results['candidates']
                    st.session_state["last_job_description"] = job_description
                    stored_candidates = results['candidates']
                    stored_job_desc = job_description
                else:
                    st.error("❌ Failed to rank candidates. Please check your configuration.")
    
    # Display stored candidates if they exist (even after reruns)
    if stored_candidates:
        # Summary metrics
        st.subheader("📊 Summary")
        avg_score = sum(c['final_score'] for c in stored_candidates) / len(stored_candidates) if stored_candidates else 0
        max_score = max((c['final_score'] for c in stored_candidates), default=0)
        min_score = min((c['final_score'] for c in stored_candidates), default=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Score", f"{avg_score:.2f}")
        with col2:
            st.metric("Highest Score", f"{max_score:.2f}")
        with col3:
            st.metric("Lowest Score", f"{min_score:.2f}")
        with col4:
            st.metric("Candidates", len(stored_candidates))
        
        st.divider()
        
        # Check all candidates for interview button clicks
        for idx, candidate in enumerate(stored_candidates, 1):
            candidate_id = candidate.get('meta', {}).get('id', '')
            interview_flag_key = f"start_interview_{candidate_id}_{idx}"
            if st.session_state.get(interview_flag_key, False):
                # Set persistent interview state
                unique_key = f"{candidate_id}_{idx}"
                st.session_state["active_interview_key"] = unique_key
                st.session_state["active_interview_candidate"] = candidate
                st.session_state["active_interview_index"] = idx
                st.session_state["active_interview_job_desc"] = stored_job_desc
                # Store candidate data for later retrieval
                st.session_state[f"candidate_data_{unique_key}"] = candidate
                # Reset the trigger flag
                st.session_state[interview_flag_key] = False
                st.rerun()
        
        # Display candidates
        if not st.session_state.get("active_interview_key"):
            st.subheader("👥 Ranked Candidates")
        else:
            st.markdown("### 👥 Ranked Candidates (Scroll down)")
        for idx, candidate in enumerate(stored_candidates, 1):
            display_candidate(candidate, idx, job_description=stored_job_desc)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #4b5563 !important; font-size: 0.875rem;'>"
        "CV Ranking System | Powered by AI & Explainable AI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

