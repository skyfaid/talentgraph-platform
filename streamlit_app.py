"""
Streamlit Frontend - Fixed Version with Live Video & Automatic Flow
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import json
from pathlib import Path
import sys
import threading
import time
import cv2
import queue
import numpy as np

# Add project root
sys.path.append(str(Path(__file__).parent))

from main import AIInterviewer
from utils.config import MIN_QUESTIONS, MAX_QUESTIONS

# Page config
st.set_page_config(page_title="AI Interviewer", page_icon="🤖", layout="wide")

# CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .score-excellent {color: #2ecc71; font-weight: bold;}
    .score-good {color: #3498db; font-weight: bold;}
    .score-moderate {color: #f39c12; font-weight: bold;}
    .score-poor {color: #e74c3c; font-weight: bold;}
    .recording-indicator {
        background-color: #e74c3c;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% {opacity: 1;}
        50% {opacity: 0.5;}
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'interview_started': False,
        'interviewer': None,
        'report': None,
        'question_num': 0,
        'current_question': "",
        'video_frame_queue': queue.Queue(maxsize=2),
        'video_thread': None,
        'video_active': False,
        'video_active_flag': None,
        'audio_recording': False,
        'audio_thread': None,
        'audio_state': {},
        'current_answer': None,
        'current_audio_path': None,
        'waiting_for_answer': False,
        'tts_playing': False,
        'processing_answer': False,
        'needs_auto_record': False,
        'needs_tts': False,
        'stop_recording_requested': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def video_capture_thread(interviewer, video_active_flag, frame_queue):
    """Background thread to continuously capture video frames"""
    
    while video_active_flag['active']:
        try:
            frame = interviewer.video_processor.get_current_frame()
            
            if frame is not None:
                # Add recording indicator
                frame_copy = frame.copy()
                cv2.circle(frame_copy, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame_copy, "REC", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add emotion overlay
                if interviewer.video_processor.emotion_history:
                    latest = interviewer.video_processor.emotion_history[-1]['emotions']
                    dominant = max(latest.items(), key=lambda x: x[1])
                    cv2.putText(frame_copy, f"{dominant[0]}: {dominant[1]:.2f}", 
                               (10, frame_copy.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue (non-blocking)
                try:
                    if not frame_queue.full():
                        frame_queue.put_nowait(frame_rgb)
                    else:
                        # Remove old frame and add new one
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame_rgb)
                        except:
                            pass
                except:
                    pass
            
            time.sleep(0.016)  # ~60 FPS (1000ms / 60 = 16.67ms)
            
        except Exception as e:
            pass
    
    pass


def audio_recording_thread(interviewer, state_dict):
    """Background thread to record audio answer
    
    Uses a shared state dict to communicate results back to main thread.
    Monitors state_dict['stop'] to allow external stopping.
    """
    
    try:
        import sounddevice as sd
        
        state_dict['answer'] = None
        state_dict['audio_path'] = None
        
        # Use record_audio but with a patched input() that checks our stop flag
        interviewer.audio_processor.recording = []
        interviewer.audio_processor.is_recording = True
        
        def audio_callback(indata, frames, time_info, status):
            if interviewer.audio_processor.is_recording:
                interviewer.audio_processor.recording.append(indata.copy())
        
        # Create audio stream
        stream = sd.InputStream(
            samplerate=interviewer.audio_processor.sample_rate,
            channels=1,
            callback=audio_callback,
            dtype='float32',
            blocksize=2048,
            latency='low'
        )
        
        with stream:
            # Wait for stop signal or max duration
            max_duration = 120
            start_time = time.time()
            
            while (time.time() - start_time) < max_duration:
                # Check if stop was requested from UI
                if state_dict.get('stop', False):
                    break
                time.sleep(0.05)  # Check every 50ms
            
            interviewer.audio_processor.is_recording = False
        
        # Save and transcribe
        if interviewer.audio_processor.recording:
            # Save audio
            import numpy as np
            audio_data = np.concatenate(interviewer.audio_processor.recording, axis=0)
            
            from scipy.io import wavfile
            from pathlib import Path
            temp_file = Path(interviewer.audio_processor.temp_dir) / f"temp_answer_{int(time.time())}.wav"
            temp_file.parent.mkdir(exist_ok=True, parents=True)
            wavfile.write(str(temp_file), interviewer.audio_processor.sample_rate, audio_data)
            
            # Transcribe
            answer = interviewer.audio_processor.transcribe_audio(str(temp_file))
            state_dict['answer'] = answer
            state_dict['audio_path'] = str(temp_file)
        else:
            state_dict['answer'] = None
            state_dict['audio_path'] = None
            
    except Exception as e:
        state_dict['answer'] = None
        state_dict['audio_path'] = None
    
    finally:
        state_dict['recording_complete'] = True


def start_video_recording():
    """Start video recording and live capture"""
    interviewer = st.session_state.interviewer
    
    # Start video recording
    output_path = str(interviewer.video_processor.temp_dir / 
                     f"interview_{st.session_state.candidate_name}_{int(time.time())}.mp4")
    interviewer.video_processor.start_recording(output_path, fps=15)
    
    # Create flag dict that thread can modify
    st.session_state.video_active_flag = {'active': True}
    st.session_state.video_active = True
    
    # Wait for camera initialization
    time.sleep(1.0)
    
    # Start background thread for frame capture
    st.session_state.video_thread = threading.Thread(
        target=video_capture_thread,
        args=(interviewer, st.session_state.video_active_flag, st.session_state.video_frame_queue),
        daemon=True
    )
    st.session_state.video_thread.start()


def stop_video_recording():
    """Stop video recording and capture thread"""
    # Stop the thread
    if hasattr(st.session_state, 'video_active_flag'):
        st.session_state.video_active_flag['active'] = False
    st.session_state.video_active = False
    
    # Wait for thread
    if st.session_state.video_thread:
        st.session_state.video_thread.join(timeout=2.0)
        st.session_state.video_thread = None
    
    # Stop video processor
    if st.session_state.interviewer and st.session_state.interviewer.video_processor:
        st.session_state.interviewer.video_processor.stop_recording()
        print("✓ Video recording stopped")


def start_audio_recording():
    """Start audio recording in background"""
    interviewer = st.session_state.interviewer
    
    st.session_state.audio_recording = True
    st.session_state.current_answer = None
    st.session_state.current_audio_path = None
    st.session_state.stop_recording_requested = False
    
    # Shared state dict for thread communication
    st.session_state.audio_state = {
        'answer': None,
        'audio_path': None,
        'recording_complete': False,
        'stop': False
    }
    
    st.session_state.audio_thread = threading.Thread(
        target=audio_recording_thread,
        args=(interviewer, st.session_state.audio_state),
        daemon=True
    )
    st.session_state.audio_thread.start()


def stop_audio_recording():
    """Stop the currently recording audio (like pressing ENTER in CLI)"""
    if st.session_state.audio_recording and 'audio_state' in st.session_state:
        st.session_state.audio_state['stop'] = True
        st.session_state.stop_recording_requested = True


def speak_question_async(question: str, interviewer):
    """Speak question in background thread"""
    def speak():
        try:
            st.session_state.tts_playing = True
            interviewer.audio_processor.speak(question)
        except Exception as e:
            pass
        finally:
            st.session_state.tts_playing = False
    
    thread = threading.Thread(target=speak, daemon=True)
    thread.start()


def main():
    init_session_state()
    
    st.markdown("<h1 class='main-header'>🤖 AI Interview System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Mode selection
        mode = st.selectbox("Interview Mode", ["Text", "Voice", "Video"], 
                           disabled=st.session_state.interview_started)
        
        # Info
        if mode == "Voice":
            st.info("🎤 Voice mode:\n• AI speaks questions automatically\n• Speak your answers\n• Press ENTER in terminal/console to stop recording")
        elif mode == "Video":
            st.info("📹 Video mode:\n• Live camera feed\n• AI speaks questions\n• Press ENTER to stop recording\n• Facial emotion analysis")
        
        # Candidate info
        st.subheader("Candidate Info")
        candidate_name = st.text_input("Name", value="Test Candidate",
                                       disabled=st.session_state.interview_started)
        
        st.subheader("Job Details")
        job_title = st.text_input("Position", value="Software Engineer",
                                  disabled=st.session_state.interview_started)
        skills_input = st.text_area("Skills", value="python, machine learning, backend development",
                                    disabled=st.session_state.interview_started)
        required_skills = [s.strip() for s in skills_input.split(',')]
        
        # Start button
        if st.button("🚀 Start Interview", type="primary", 
                    disabled=st.session_state.interview_started):
            st.session_state.interview_started = True
            st.session_state.mode = mode.lower()
            st.session_state.candidate_name = candidate_name
            st.session_state.job_context = {'title': job_title, 'skills': required_skills}
            
            # Initialize AIInterviewer
            st.session_state.interviewer = AIInterviewer(mode=st.session_state.mode)
            st.session_state.interviewer.interview_data['candidate_name'] = candidate_name
            
            # Generate first question
            st.session_state.current_question = st.session_state.interviewer.question_generator.generate_question(
                job_context=st.session_state.job_context
            )
            st.session_state.question_num = 1
            st.session_state.waiting_for_answer = True
            
            # Start video recording ONCE at the beginning (continuous for entire interview)
            if st.session_state.mode == 'video':
                start_video_recording()
            
            # Mark that TTS should happen after page renders
            st.session_state.needs_tts = True
            
            st.rerun()
        
        # Reset
        if st.button("🔄 Reset"):
            if st.session_state.video_active:
                stop_video_recording()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if not st.session_state.interview_started:
        show_welcome()
    elif st.session_state.report:
        show_results()
    else:
        show_interview()


def show_welcome():
    """Welcome page"""
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### 📝 Text Mode")
        st.write("• Type your answers\n• Text sentiment analysis\n• Quick and simple")
    
    with cols[1]:
        st.markdown("### 🎤 Voice Mode")
        st.write("• Speak your answers\n• Voice emotion detection\n• Press ENTER to stop")
    
    with cols[2]:
        st.markdown("### 📹 Video Mode")
        st.write("• Video + audio recording\n• Facial emotion analysis\n• Complete assessment")
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.write("1. Select interview mode\n2. Enter candidate details\n3. Click 'Start Interview'")
    st.write("\n**Note:** In Voice/Video modes, recording stops when you press ENTER in your terminal/console window.")


def show_interview():
    """Main interview interface with automatic flow"""
    
    # Handle deferred TTS (after page has rendered)
    if st.session_state.get('needs_tts', False):
        st.session_state.needs_tts = False
        if st.session_state.mode in ['voice', 'video']:
            speak_question_async(st.session_state.current_question, st.session_state.interviewer)
            st.session_state.needs_auto_record = True
    
    st.markdown(f"### 🎯 Question {st.session_state.question_num}/{MAX_QUESTIONS}")
    
    # Progress
    progress = st.session_state.question_num / MAX_QUESTIONS
    st.progress(progress)
    
    # Display question
    st.markdown("#### 💬 Question:")
    st.info(st.session_state.current_question)
    
    # Mode-specific interface
    if st.session_state.mode == 'text':
        handle_text_mode()
    elif st.session_state.mode == 'voice':
        handle_voice_mode_auto()
    elif st.session_state.mode == 'video':
        handle_video_mode_auto()
    
    # Previous Q&A
    if st.session_state.interviewer.interview_data['qa_pairs']:
        st.markdown("---")
        with st.expander(f"📋 Previous {len(st.session_state.interviewer.interview_data['qa_pairs'])} Answers"):
            for i, qa in enumerate(st.session_state.interviewer.interview_data['qa_pairs'], 1):
                st.markdown(f"**Q{i}:** {qa['question'][:60]}...")
                st.markdown(f"**A{i}:** {qa['answer'][:80]}...")


def handle_text_mode():
    """Text mode - simple input"""
    answer = st.text_area("Your Answer:", height=150, 
                         key=f"text_{st.session_state.question_num}",
                         disabled=st.session_state.processing_answer)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Submit Answer", type="primary", 
                    disabled=st.session_state.processing_answer):
            if answer.strip():
                process_answer(answer)
            else:
                st.error("Please provide an answer")
    
    with col2:
        if st.session_state.question_num >= MIN_QUESTIONS:
            if st.button("End Interview"):
                complete_interview()


def handle_voice_mode_auto():
    """Voice mode - automatic flow like CLI"""
    
    # Check if recording thread has completed and update session state
    if (st.session_state.audio_recording and 
        'audio_state' in st.session_state and
        st.session_state.audio_state.get('recording_complete', False)):
        # Recording finished - copy results to session state
        st.session_state.current_answer = st.session_state.audio_state.get('answer')
        st.session_state.current_audio_path = st.session_state.audio_state.get('audio_path')
        st.session_state.audio_recording = False
    
    # Auto-start recording if TTS finished and not yet recording
    if (st.session_state.get('needs_auto_record', False) and 
        not st.session_state.tts_playing and 
        not st.session_state.audio_recording and
        st.session_state.current_answer is None):
        st.session_state.needs_auto_record = False
        start_audio_recording()
        time.sleep(0.5)
        st.rerun()
    
    # Show TTS status
    if st.session_state.tts_playing:
        st.info("🔊 AI is speaking the question...")
        time.sleep(0.5)
        st.rerun()
    
    # Show recording status
    elif st.session_state.audio_recording:
        st.markdown("<div class='recording-indicator'>🎤 RECORDING...</div>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 **Speak your answer now...**")
        with col2:
            if st.button("⏹️ Stop & Submit", type="primary", key="voice_stop_btn"):
                stop_audio_recording()
                time.sleep(0.5)
                st.rerun()
        
        # Auto-refresh to check if recording finished
        time.sleep(0.3)
        st.rerun()
    
    # Check if answer is ready
    elif st.session_state.current_answer:
        st.success(f"📝 Transcribed: {st.session_state.current_answer}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("✅ Confirm & Submit", type="primary"):
                answer = st.session_state.current_answer
                audio_path = st.session_state.get('current_audio_path')
                process_answer(answer, audio_path)
        
        with col2:
            if st.button("🔄 Re-record"):
                st.session_state.current_answer = None
                st.session_state.current_audio_path = None
                start_audio_recording()
                st.rerun()
    
    # Waiting to start recording
    elif st.session_state.waiting_for_answer and not st.session_state.tts_playing:
        if st.session_state.get('needs_auto_record', False):
            st.warning("⏳ Starting recording...")
            time.sleep(0.5)
            st.rerun()
        else:
            st.success("✅ Ready for next question")
    
    # Text fallback
    st.markdown("---")
    st.markdown("*Or type your answer:*")
    text_answer = st.text_input("Type answer:", key=f"voice_text_{st.session_state.question_num}")
    
    if text_answer and st.button("Submit Typed Answer"):
        process_answer(text_answer)
    
    # End interview option
    if st.session_state.question_num >= MIN_QUESTIONS:
        st.markdown("---")
        if st.button("🏁 End Interview"):
            complete_interview()


def handle_video_mode_auto():
    """Video mode - automatic flow with live video"""
    
    # Check if recording thread has completed and update session state
    if (st.session_state.audio_recording and 
        'audio_state' in st.session_state and
        st.session_state.audio_state.get('recording_complete', False)):
        # Recording finished - copy results to session state
        st.session_state.current_answer = st.session_state.audio_state.get('answer')
        st.session_state.current_audio_path = st.session_state.audio_state.get('audio_path')
        st.session_state.audio_recording = False
    
    # Auto-start recording if TTS finished and not yet recording
    if (st.session_state.get('needs_auto_record', False) and 
        not st.session_state.tts_playing and 
        not st.session_state.audio_recording and
        st.session_state.current_answer is None):
        st.session_state.needs_auto_record = False
        start_audio_recording()
        time.sleep(0.3)
        # Don't return here, continue to show video
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📹 Live Video Feed")
        video_placeholder = st.empty()
        
        # Get latest frame from queue
        frame = None
        try:
            frame = st.session_state.video_frame_queue.get_nowait()
        except queue.Empty:
            # No new frame, try to get current frame directly
            if st.session_state.interviewer and st.session_state.interviewer.video_processor:
                frame = st.session_state.interviewer.video_processor.get_current_frame()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only display if we have a frame
        if frame is not None:
            video_placeholder.image(frame, channels="RGB", width='stretch', use_column_width=True)
        else:
            video_placeholder.info("📹 Loading video feed...")
    
    with col2:
        st.markdown("#### 🎭 Live Emotions")
        
        if st.session_state.video_active and st.session_state.interviewer.video_processor.emotion_history:
            latest = st.session_state.interviewer.video_processor.emotion_history[-1]['emotions']
            dominant = max(latest.items(), key=lambda x: x[1])
            
            st.metric("Dominant", dominant[0].title(), f"{dominant[1]:.1%}")
            
            st.markdown("**All Emotions:**")
            for emotion, score in sorted(latest.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.progress(score, text=f"{emotion.title()}: {score:.1%}")
        else:
            st.info("Analyzing expressions...")
    
    st.markdown("---")
    
    # Show TTS status
    if st.session_state.tts_playing:
        st.info("🔊 AI is speaking the question...")
    
    # Show recording status
    elif st.session_state.audio_recording:
        st.markdown("<div class='recording-indicator'>🎤 RECORDING...</div>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 **Speak your answer now...**")
        with col2:
            if st.button("⏹️ Stop & Submit", type="primary", key="video_stop_btn"):
                stop_audio_recording()
                time.sleep(0.5)
                st.rerun()
    
    # Answer ready
    elif st.session_state.current_answer:
        st.success(f"📝 Transcribed: {st.session_state.current_answer}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("✅ Confirm & Submit", type="primary"):
                answer = st.session_state.current_answer
                audio_path = st.session_state.get('current_audio_path')
                process_answer(answer, audio_path)
        
        with col2:
            if st.button("🔄 Re-record"):
                st.session_state.current_answer = None
                st.session_state.current_audio_path = None
                start_audio_recording()
                st.rerun()
    
    # Waiting
    elif st.session_state.waiting_for_answer and not st.session_state.tts_playing:
        if st.session_state.get('needs_auto_record', False):
            st.warning("⏳ Starting recording...")
        else:
            st.success("✅ Ready for next question")
    
    # Text fallback
    st.markdown("---")
    st.markdown("*Or type:*")
    text_answer = st.text_input("Type:", key=f"video_text_{st.session_state.question_num}")
    
    if text_answer and st.button("Submit Typed"):
        process_answer(text_answer)
    
    # End interview
    if st.session_state.question_num >= MIN_QUESTIONS:
        st.markdown("---")
        if st.button("🏁 End Interview"):
            stop_video_recording()
            complete_interview()
    
    # Keep video updating (only if recording or showing video)
    if st.session_state.audio_recording or st.session_state.tts_playing or st.session_state.get('needs_auto_record', False):
        time.sleep(0.1)
        st.rerun()


def process_answer(answer: str, audio_path: str = None):
    """Process answer using main.py's _analyze_answer"""
    st.session_state.processing_answer = True
    interviewer = st.session_state.interviewer
    
    with st.spinner("Analyzing your answer..."):
        interviewer._analyze_answer(
            st.session_state.current_question,
            answer,
            audio_path,
            st.session_state.job_context
        )
    
    st.success("✓ Answer analyzed")
    
    # Reset answer state
    st.session_state.current_answer = None
    st.session_state.current_audio_path = None
    st.session_state.waiting_for_answer = False
    st.session_state.processing_answer = False
    
    # Check if should continue
    if st.session_state.question_num >= MAX_QUESTIONS:
        complete_interview()
        return
    
    # Generate next question
    st.session_state.question_num += 1
    prev_answer = interviewer.interview_data['qa_pairs'][-1]['answer']
    st.session_state.current_question = interviewer.question_generator.generate_question(
        prev_answer,
        st.session_state.job_context
    )
    st.session_state.waiting_for_answer = True
    
    # Auto-speak and start recording for voice/video modes
    # Video recording continues (not stopped/restarted)
    if st.session_state.mode in ['voice', 'video']:
        st.session_state.needs_tts = True  # Defer TTS until page renders
        st.session_state.needs_auto_record = True
    
    time.sleep(0.5)
    st.rerun()


def complete_interview():
    """Complete interview and generate report"""
    interviewer = st.session_state.interviewer
    
    # Stop video only at the very end (continuous recording throughout)
    if st.session_state.video_active:
        stop_video_recording()
        
        # Get aggregated video emotions
        video_emotions = interviewer.video_processor.get_aggregated_emotions()
        interviewer.interview_data['video_emotion'] = video_emotions
    
    # Calculate duration
    interviewer.interview_data['duration'] = st.session_state.question_num * 2
    
    # Generate report
    with st.spinner("Generating comprehensive report..."):
        scores = interviewer.scoring_system.calculate_final_score(interviewer.interview_data)
        report = interviewer.scoring_system.generate_report(interviewer.interview_data, scores)
    
    st.session_state.report = report
    interviewer.scoring_system.save_report(report)
    
    st.rerun()


def show_results():
    """Display results"""
    st.markdown("## 📊 Interview Results")
    
    report = st.session_state.report
    score = report['scores']['overall_score']
    
    # Overall score
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.metric("Overall Score", f"{score:.1f}/10")
    
    with col2:
        score_class = "score-excellent" if score >= 8 else "score-good" if score >= 6.5 else "score-moderate" if score >= 5 else "score-poor"
        st.markdown(f"<p class='{score_class}'>{report['recommendation']}</p>", unsafe_allow_html=True)
    
    with col3:
        st.write(report['summary'])
    
    # Component scores
    st.markdown("---")
    st.markdown("### 📈 Component Scores")
    
    cols = st.columns(5)
    scores = report['scores']
    
    for col, (name, val) in zip(cols, [
        ("Motivation", scores['motivation']),
        ("Experience", scores['experience_fit']),
        ("Confidence", scores['confidence']),
        ("Logistics", scores['logistics']),
        ("Red Flags", scores['red_flags_score'])
    ]):
        col.metric(name, f"{val:.1f}/10")
    
    # Feedback
    if 'feedback' in report and report['feedback']:
        st.markdown("---")
        st.markdown("### 💡 Feedback")
        
        feedback = report['feedback']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Strengths")
            for strength in feedback.get('strengths', []):
                st.success(strength)
        
        with col2:
            st.markdown("#### 📈 Improvements")
            for area in feedback.get('areas_for_improvement', []):
                st.warning(area)
    
    # Download
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        report_json = json.dumps(report, indent=2)
        st.download_button("📥 Download Report", report_json,
                          file_name=f"interview_{report['metadata']['candidate_name']}.json")
    
    with col2:
        if st.button("🔄 New Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()