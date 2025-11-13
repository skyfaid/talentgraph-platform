"""
NiceGUI Frontend for AI Interviewer System
Full-featured implementation with video, voice, and XAI support
Run with: python nicegui_app.py
"""

from nicegui import ui, app
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np
from datetime import datetime
import json
import queue
from typing import Optional, Dict, List
import base64
import asyncio
# Add project root
sys.path.append(str(Path(__file__).parent))

from main import AIInterviewer
from utils.config import MIN_QUESTIONS, MAX_QUESTIONS, SCORING_WEIGHTS

# Global state management
class InterviewState:
    def __init__(self):
        self.interview_started = False
        self.interviewer: Optional[AIInterviewer] = None
        self.mode = 'text'
        self.candidate_name = ''
        self.job_context = {}
        self.question_num = 0
        self.current_question = ''
        self.current_answer = None
        self.current_audio_path = None
        self.report = None
        
        # Video state
        self.video_active = False
        self.video_thread = None
        self.video_frame_queue = queue.Queue(maxsize=2)
        self.video_active_flag = {'active': False}
        
        # Audio state
        self.audio_recording = False
        self.audio_thread = None
        self.audio_state = {}
        self.stop_recording_requested = False
        
        # UI state
        self.tts_playing = False
        self.processing_answer = False
        self.waiting_for_answer = False
        self.needs_tts = False
        self.needs_auto_record = False

state = InterviewState()


# =============================================================================
# BACKGROUND THREADS
# =============================================================================

def video_capture_thread():
    """Continuous video frame capture at stable FPS"""
    print("✓ Video capture thread started")
    frame_count = 0
    
    while state.video_active_flag['active']:
        try:
            frame = state.interviewer.video_processor.get_current_frame()
            
            if frame is not None:
                frame_count += 1
                
                # Create a clean copy for display
                display_frame = frame.copy()
                
                # Add recording indicator
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add emotion overlay if available
                if state.interviewer.video_processor.emotion_history:
                    latest = state.interviewer.video_processor.emotion_history[-1]['emotions']
                    dominant = max(latest.items(), key=lambda x: x[1])
                    cv2.putText(display_frame, f"{dominant[0]}: {dominant[1]:.2f}", 
                               (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # FIX: OpenCV captures in BGR, but get_current_frame might already be RGB
                # Only convert if it's actually BGR (check by comparing channels)
                # Simple heuristic: if blue channel is much higher than red, it's likely BGR
                if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                    # Check if it looks like BGR (blue dominant when should be red/skin tones)
                    # For most webcam frames, R and G should be similar or R > B for skin
                    # If B > R significantly, it's probably BGR that needs conversion
                    mean_channels = display_frame.mean(axis=(0, 1))
                    if mean_channels[0] > mean_channels[2]:  # Blue > Red = BGR format
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    else:
                        # Already RGB or close enough
                        display_frame_rgb = display_frame
                else:
                    display_frame_rgb = display_frame
                
                # Update queue with frame rate limiting
                try:
                    # Clear old frames to prevent lag
                    while state.video_frame_queue.qsize() >= 1:
                        state.video_frame_queue.get_nowait()
                    
                    state.video_frame_queue.put_nowait(display_frame_rgb)
                except:
                    pass
            
            # FIX: Slower frame rate to prevent flashing (30ms = ~33 FPS, too fast for web)
            time.sleep(0.1)  # 100ms = 10 FPS, much more stable for web display
            
        except Exception as e:
            if frame_count < 10:
                print(f"❌ Video error: {e}")
            pass
    
    print(f"✓ Video thread stopped ({frame_count} frames)")


def audio_recording_thread():
    """Record audio in background with stop control"""
    try:
        import pyaudio
        import wave
        
        print("🎤 Audio recording started")
        
        state.audio_state['answer'] = None
        state.audio_state['audio_path'] = None
        state.audio_state['recording_complete'] = False
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000  # Standard for Whisper
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                       input=True, frames_per_buffer=CHUNK)
        
        frames = []
        start_time = time.time()
        max_duration = 120
        
        # Record until stop flag or timeout
        while (time.time() - start_time) < max_duration:
            if state.audio_state.get('stop', False):
                print("🛑 Recording stop requested")
                break
            
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except:
                break
            
            time.sleep(0.01)
        
        duration = (time.time() - start_time)
        print(f"✓ Recording finished ({len(frames)} chunks, ~{duration:.1f}s)")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save and transcribe
        if frames and len(frames) > 10:  # At least 0.5 seconds
            temp_dir = Path("data/temp_audio")
            temp_dir.mkdir(exist_ok=True, parents=True)
            audio_path = temp_dir / f"temp_answer_{int(time.time())}.wav"
            
            wf = wave.open(str(audio_path), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"✓ Audio saved: {audio_path} ({duration:.1f}s)")
            
            # FIX: Use the WORKING transcribe_audio method from your AudioProcessor
            try:
                print("🔄 Transcribing with Whisper (using AudioProcessor method)...")
                
                # THIS is the key fix - call YOUR working method directly
                answer = state.interviewer.audio_processor.transcribe_audio(str(audio_path))
                
                if answer and len(answer.strip()) > 0:
                    print(f"✓ Transcribed ({len(answer)} chars): {answer}")
                    state.audio_state['answer'] = answer
                    state.audio_state['audio_path'] = str(audio_path)
                else:
                    print("⚠️ Transcription returned empty")
                    state.audio_state['answer'] = None
                    state.audio_state['audio_path'] = None
                    
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                import traceback
                traceback.print_exc()
                state.audio_state['answer'] = None
                state.audio_state['audio_path'] = None
        else:
            print("⚠️ No audio recorded (too short)")
            state.audio_state['answer'] = None
            state.audio_state['audio_path'] = None
            
    except Exception as e:
        print(f"❌ Audio recording error: {e}")
        import traceback
        traceback.print_exc()
        state.audio_state['answer'] = None
        state.audio_state['audio_path'] = None
    
    finally:
        state.audio_state['recording_complete'] = True
        print("✓ Audio recording complete")


def tts_thread(text: str):
    """Text-to-speech in background with robust error handling"""
    try:
        print(f"🔊 Speaking: {text}")
        state.tts_playing = True
        
        # Add a small delay to ensure previous TTS is cleared
        time.sleep(0.5)
        
        # Use a simpler approach - just speak the text
        state.interviewer.audio_processor.speak(text)
        
        print("✓ TTS finished")
    except Exception as e:
        print(f"❌ TTS error: {e}")
        # Try alternative TTS method as fallback
        try:
            print("🔄 Trying alternative TTS method...")
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            print("✓ Fallback TTS finished")
        except Exception as e2:
            print(f"❌ Fallback TTS also failed: {e2}")
    finally:
        state.tts_playing = False


# =============================================================================
# VIDEO/AUDIO CONTROL FUNCTIONS
# =============================================================================

def start_video_recording():
    """Start continuous video recording"""
    output_path = str(state.interviewer.video_processor.temp_dir / 
                     f"interview_{state.candidate_name}_{int(time.time())}.mp4")
    
    print(f"🎥 Starting video recording to: {output_path}")
    
    # Test camera before starting
    try:
        test_frame = state.interviewer.video_processor.get_current_frame()
        if test_frame is not None:
            print(f"✅ Camera test successful - Frame: {test_frame.shape}")
        else:
            print("❌ Camera test failed - no frame received")
    except Exception as e:
        print(f"❌ Camera test error: {e}")
    
    state.interviewer.video_processor.start_recording(output_path, fps=15)
    
    state.video_active_flag['active'] = True
    state.video_active = True
    
    time.sleep(1.0)  # Camera init
    
    state.video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    state.video_thread.start()
    print("✓ Video recording started")


def stop_video_recording():
    """Stop video recording"""
    state.video_active_flag['active'] = False
    state.video_active = False
    
    if state.video_thread:
        state.video_thread.join(timeout=2.0)
        state.video_thread = None
    
    if state.interviewer and state.interviewer.video_processor:
        state.interviewer.video_processor.stop_recording()
        print("✓ Video recording stopped")


def start_audio_recording():
    """Start audio recording"""
    state.audio_recording = True
    state.current_answer = None
    state.current_audio_path = None
    
    state.audio_state = {
        'answer': None,
        'audio_path': None,
        'recording_complete': False,
        'stop': False
    }
    
    state.audio_thread = threading.Thread(target=audio_recording_thread, daemon=True)
    state.audio_thread.start()
    print("✓ Audio recording started")


def stop_audio_recording():
    """Stop audio recording"""
    if state.audio_recording and state.audio_state:
        state.audio_state['stop'] = True
        print("🛑 Audio stop requested")


def speak_question(question: str):
    """Speak question asynchronously with better timing"""
    # Don't start new TTS if one is already running
    if state.tts_playing:
        print("⚠️ TTS already in progress, skipping...")
        return
    
    thread = threading.Thread(target=tts_thread, args=(question,), daemon=True)
    thread.start()


# =============================================================================
# INTERVIEW LOGIC
# =============================================================================

def start_interview(name: str, job_title: str, skills: str, mode: str):
    """Initialize and start interview"""
    state.interview_started = True
    state.mode = mode
    state.candidate_name = name
    state.job_context = {
        'title': job_title,
        'skills': [s.strip() for s in skills.split(',')]
    }
    
    # Initialize interviewer
    state.interviewer = AIInterviewer(mode=state.mode)
    state.interviewer.interview_data['candidate_name'] = name
    
    # Generate first question
    state.current_question = state.interviewer.question_generator.generate_question(
        job_context=state.job_context
    )
    state.question_num = 1
    state.waiting_for_answer = True
    
    # Start video if needed
    if state.mode == 'video':
        start_video_recording()
    
    # Speak question for voice/video modes
    if state.mode in ['voice', 'video']:
        state.needs_tts = True
    
    print(f"✓ Interview started: {name} for {job_title}")


def process_answer(answer: str, audio_path: str = None):
    """Process candidate answer with better error handling"""
    print(f"🔄 Processing answer: {answer[:50]}...")
    state.processing_answer = True
    
    try:
        # Analyze answer
        print("🔄 Starting answer analysis...")
        analysis_result = state.interviewer._analyze_answer(
            state.current_question,
            answer,
            audio_path,
            state.job_context
        )
        print("✅ Answer analysis completed")
        
        # Check if interview should end
        if state.question_num >= MAX_QUESTIONS:
            print("🏁 Interview completed - generating report")
            complete_interview()
            # Reset state for next question
            state.current_answer = None
            state.current_audio_path = None
            state.waiting_for_answer = False
            state.processing_answer = False
            return
        
        # Generate next question
        print("🔄 Generating next question...")
        state.question_num += 1
        
        # Get the previous answer for context
        prev_answer = state.interviewer.interview_data['qa_pairs'][-1]['answer'] if state.interviewer.interview_data['qa_pairs'] else ""
        
        state.current_question = state.interviewer.question_generator.generate_question(
            prev_answer,
            state.job_context
        )
        state.waiting_for_answer = True
        
        print(f"✅ Question {state.question_num} ready: {state.current_question[:50]}...")
        
        # Speak next question for voice/video modes
        if state.mode in ['voice', 'video']:
            state.needs_tts = True
            state.needs_auto_record = True
            
    except Exception as e:
        print(f"❌ Error processing answer: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always reset these states
        state.current_answer = None
        state.current_audio_path = None
        state.waiting_for_answer = False
        state.processing_answer = False


def complete_interview():
    """Complete interview and generate report"""
    # Stop video
    if state.video_active:
        stop_video_recording()
        video_emotions = state.interviewer.video_processor.get_aggregated_emotions()
        state.interviewer.interview_data['video_emotion'] = video_emotions
    
    # Calculate duration
    state.interviewer.interview_data['duration'] = state.question_num * 2
    
    # Generate report
    scores = state.interviewer.scoring_system.calculate_final_score(
        state.interviewer.interview_data
    )
    report = state.interviewer.scoring_system.generate_report(
        state.interviewer.interview_data, scores
    )
    
    state.report = report
    state.interviewer.scoring_system.save_report(report)
    
    print("✓ Interview completed")


def reset_interview():
    """Reset all state"""
    if state.video_active:
        stop_video_recording()
    
    state.__init__()
    print("✓ State reset")


# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_header():
    """Create header with navigation"""
    with ui.header().classes('items-center justify-between'):
        ui.label('🤖 AI Interviewer').classes('text-2xl font-bold')
        
        with ui.row():
            ui.button(
                icon='home',
                on_click=lambda: ui.navigate.to('/')
            ).props('flat')
            
            # Dark mode toggle
            dark_mode = ui.dark_mode()
            ui.button(
                icon='dark_mode' if not dark_mode.value else 'light_mode',
                on_click=dark_mode.toggle
            ).props('flat')

def create_footer():
    """Create footer"""
    with ui.footer().classes('bg-gray-800 text-white'):
        ui.label('AI Interview System v1.0 | Powered by NiceGUI').classes('text-center w-full')

def create_welcome_page():
    """Welcome/setup page"""
    create_header()
    
    with ui.column().classes('w-full items-center'):
        ui.label('🤖 AI Interview System').classes('text-4xl font-bold text-blue-600 mb-8')
        
        with ui.card().classes('w-full max-w-4xl p-8'):
            ui.label('Select Interview Mode').classes('text-2xl font-semibold mb-4')
            
            with ui.row().classes('w-full gap-4 mb-8'):
                with ui.card().classes('flex-1 p-6 hover:shadow-lg cursor-pointer'):
                    ui.label('📝 Text Mode').classes('text-xl font-bold mb-2')
                    ui.label('• Type your answers\n• Text sentiment analysis\n• Quick and simple')
                
                with ui.card().classes('flex-1 p-6 hover:shadow-lg cursor-pointer'):
                    ui.label('🎤 Voice Mode').classes('text-xl font-bold mb-2')
                    ui.label('• Speak your answers\n• Voice emotion detection\n• Click Stop button')
                
                with ui.card().classes('flex-1 p-6 hover:shadow-lg cursor-pointer'):
                    ui.label('📹 Video Mode').classes('text-xl font-bold mb-2')
                    ui.label('• Video + audio recording\n• Facial emotion analysis\n• Complete assessment')
            
            ui.separator()
            
            with ui.row().classes('w-full gap-4 items-end'):
                mode_select = ui.select(
                    ['text', 'voice', 'video'],
                    label='Interview Mode',
                    value='text'
                ).classes('flex-1')
                
            with ui.row().classes('w-full gap-4 mt-4'):
                name_input = ui.input(
                    label='Candidate Name',
                    placeholder='Enter name',
                    value='Test Candidate'
                ).classes('flex-1')
                
                job_input = ui.input(
                    label='Job Title',
                    placeholder='e.g., Software Engineer',
                    value='Software Engineer'
                ).classes('flex-1')
            
            skills_input = ui.textarea(
                label='Required Skills (comma-separated)',
                placeholder='e.g., python, machine learning, backend',
                value='python, machine learning, backend development'
            ).classes('w-full')
            
            with ui.row().classes('w-full gap-4 mt-6'):
                ui.button(
                    '🚀 Start Interview',
                    
                    on_click=lambda: on_start_interview(
                        name_input.value,
                        job_input.value,
                        skills_input.value,
                        mode_select.value
                    )
                ).props('size=lg color=primary')
                
                ui.button(
                    '🔄 Reset',
                    on_click=lambda: (reset_interview(), ui.navigate.to('/'))
                ).props('size=lg color=secondary')
    
    create_footer()


def create_interview_page():
    """Main interview interface"""
    create_header()
    
    with ui.column().classes('w-full items-center p-4'):
        # Header
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label(f'Interview: {state.candidate_name}').classes('text-2xl font-bold')
            ui.label(f'Q{state.question_num}/{MAX_QUESTIONS}').classes('text-xl')
        
        # Progress bar
        progress = ui.linear_progress(value=state.question_num / MAX_QUESTIONS).classes('w-full mb-4')
        
        # Question display
        with ui.card().classes('w-full max-w-4xl p-6 mb-4'):
            ui.label('💬 Question:').classes('text-lg font-semibold mb-2')
            question_label = ui.label(state.current_question).classes('text-lg')
        
        # Mode-specific interface
        if state.mode == 'text':
            create_text_interface()
        elif state.mode == 'voice':
            create_voice_interface()
        elif state.mode == 'video':
            create_video_interface()
        
        # Previous Q&A
        if state.interviewer and state.interviewer.interview_data['qa_pairs']:
            with ui.expansion('📋 Previous Answers', icon='history').classes('w-full max-w-4xl'):
                for i, qa in enumerate(state.interviewer.interview_data['qa_pairs'], 1):
                    with ui.card().classes('mb-2 p-3'):
                        ui.label(f"Q{i}: {qa['question'][:80]}...").classes('font-semibold')
                        ui.label(f"A{i}: {qa['answer'][:100]}...").classes('text-gray-700')
    
    create_footer()


def create_text_interface():
    """Text mode interface"""
    with ui.card().classes('w-full max-w-4xl p-6'):
        answer_input = ui.textarea(
            label='Your Answer:',
            placeholder='Type your answer here...'
        ).classes('w-full').props('rows=8')
        
        with ui.row().classes('gap-4 mt-4'):
            ui.button(
                '✅ Submit Answer',
                on_click=lambda: (
                    process_answer(answer_input.value) if answer_input.value.strip() 
                    else ui.notify('Please provide an answer', type='warning'),
                    ui.navigate.reload()
                )
            ).props('color=primary')
            
            if state.question_num >= MIN_QUESTIONS:
                ui.button(
                    '🏁 End Interview',
                    on_click=lambda: (complete_interview(), ui.navigate.to('/results'))
                ).props('color=secondary')


def create_voice_interface():
    """Voice mode interface"""
    with ui.card().classes('w-full max-w-4xl p-6'):
        status_label = ui.label('').classes('text-lg mb-4')
        
        def update_status():
            if state.tts_playing:
                status_label.text = '🔊 AI is speaking...'
                status_label.classes('text-blue-600')
            elif state.audio_recording:
                status_label.text = '🎤 RECORDING... Speak now! (Click "Stop Recording" when done)'
                status_label.classes('text-red-600 font-bold')
            elif state.current_answer:
                status_label.text = f'📝 Transcribed: {state.current_answer}'
                status_label.classes('text-green-600')
            else:
                status_label.text = '⏳ Ready - Click "Start Recording" to begin'
                status_label.classes('text-gray-600')
        
        update_status()
        
        with ui.row().classes('gap-4'):
            if state.audio_recording:
                # Stop recording button
                def stop_and_process():
                    stop_audio_recording()
                    time.sleep(2)
                    ui.navigate.reload()
                
                ui.button('⏹️ Stop Recording', on_click=stop_and_process) \
                  .props('color=negative size=lg')
                
            elif not state.audio_recording and not state.tts_playing and state.waiting_for_answer:
                # Start recording button
                def start_recording():
                    start_audio_recording()
                    ui.navigate.reload()
                
                ui.button('🎤 Start Recording', on_click=start_recording) \
                  .props('color=primary size=lg')
            
            elif state.current_answer:
                # Submit answer button
                def submit_answer():
                    print(f"🔄 Submit button clicked - Answer: {state.current_answer[:50]}...")
                    if state.current_answer and state.current_answer.strip():
                        # Process in thread to avoid blocking UI
                        def process_and_reload():
                            process_answer(state.current_answer, state.current_audio_path)
                            # Small delay before reload to ensure processing completes
                            time.sleep(1)
                            ui.navigate.reload()
                        
                        threading.Thread(target=process_and_reload, daemon=True).start()
                        ui.notify('Processing your answer...', type='info')
                    else:
                        ui.notify('No answer to submit', type='warning')
                
                ui.button('✅ Submit Answer', on_click=submit_answer) \
                .props('color=positive size=lg')
                            
                # Re-record button
                def rerecord():
                    state.current_answer = None
                    state.current_audio_path = None
                    start_audio_recording()
                    ui.navigate.reload()
                
                ui.button('🔄 Re-record', on_click=rerecord) \
                  .props('color=secondary')
        
        # Text fallback
        ui.separator().classes('my-4')
        ui.label('Or type your answer:').classes('text-sm text-gray-600')
        text_input = ui.input(placeholder='Type answer...').classes('w-full')
        
        def submit_text_answer():
            if text_input.value.strip():
                process_answer(text_input.value)
                ui.navigate.reload()
            else:
                ui.notify('Please provide an answer', type='warning')
        
        ui.button('Submit Typed Answer', on_click=submit_text_answer)
        
        if state.question_num >= MIN_QUESTIONS:
            ui.separator().classes('my-4')
            ui.button('🏁 End Interview', 
                     on_click=lambda: (complete_interview(), ui.navigate.to('/results'))) \
              .props('color=secondary')


def create_video_interface():
    """Video mode interface with live feed"""
    with ui.row().classes('w-full max-w-6xl gap-4'):
        # Video feed column
        with ui.card().classes('flex-1 p-4'):
            ui.label('📹 Live Video Feed').classes('text-lg font-semibold mb-2')
            video_image = ui.image('').classes('w-full rounded border-2 max-h-96 object-cover')
            
            # FIX: Use a slower timer with proper error handling to prevent flashing
            last_update_time = {'time': 0}  # Store in dict to modify in nested function
            
            def update_video():
                try:
                    current_time = time.time()
                    
                    # FIX: Only update every 150ms minimum (debouncing)
                    if current_time - last_update_time['time'] < 0.15:
                        return
                    
                    if state.video_active and not state.video_frame_queue.empty():
                        frame = state.video_frame_queue.get_nowait()
                        if frame is not None:
                            # Resize for consistent display
                            frame_resized = cv2.resize(frame, (640, 480))
                            
                            # FIX: Better JPEG encoding with error handling
                            success, buffer = cv2.imencode(
                                '.jpg', 
                                frame_resized, 
                                [cv2.IMWRITE_JPEG_QUALITY, 85]  # Higher quality
                            )
                            
                            if success:
                                b64 = base64.b64encode(buffer).decode()
                                # FIX: Only update if we have valid data
                                if len(b64) > 100:  # Sanity check
                                    video_image.set_source(f'data:image/jpeg;base64,{b64}')
                                    last_update_time['time'] = current_time
                except Exception as e:
                    # Silently ignore errors to prevent log spam
                    pass
            
            if state.video_active:
                # FIX: Match the slower capture rate (150ms)
                ui.timer(0.15, update_video, active=True)
        
        # Emotions column
        with ui.card().classes('w-80 p-4'):
            ui.label('🎭 Live Emotions').classes('text-lg font-semibold mb-4')
            emotion_container = ui.column().classes('w-full')
            
            def update_emotions():
                emotion_container.clear()
                if state.video_active and state.interviewer.video_processor.emotion_history:
                    with emotion_container:
                        latest = state.interviewer.video_processor.emotion_history[-1]['emotions']
                        dominant = max(latest.items(), key=lambda x: x[1])
                        
                        ui.label(f'Dominant: {dominant[0].title()}').classes('font-bold text-lg')
                        ui.label(f'{dominant[1]:.1%}').classes('text-2xl text-blue-600')
                        
                        ui.separator().classes('my-2')
                        
                        for emotion, score in sorted(latest.items(), key=lambda x: x[1], reverse=True)[:5]:
                            with ui.row().classes('w-full items-center gap-2'):
                                ui.label(emotion.title()).classes('w-24')
                                ui.linear_progress(value=score).classes('flex-1')
                                ui.label(f'{score:.0%}').classes('w-12 text-right')
            
            if state.video_active:
                ui.timer(3.0, update_emotions, active=True)
    
    # Audio controls
    with ui.card().classes('w-full max-w-6xl p-6 mt-4'):
        status_label = ui.label('').classes('text-lg mb-4')
        
        def update_status():
            if state.tts_playing:
                status_label.text = '🔊 AI is speaking...'
                status_label.classes('text-blue-600')
            elif state.audio_recording:
                status_label.text = '🎤 RECORDING... Speak now! (Click "Stop Recording" when done)'
                status_label.classes('text-red-600 font-bold')
            elif state.current_answer:
                status_label.text = f'📝 Transcribed: {state.current_answer[:100]}...'
                status_label.classes('text-green-600')
            else:
                status_label.text = '⏳ Ready - Click "Start Recording" to begin'
                status_label.classes('text-gray-600')
        
        update_status()
        
        with ui.row().classes('gap-4'):
            if state.audio_recording:
                # Stop recording button
                def stop_and_process():
                    stop_audio_recording()
                    # Wait for processing
                    time.sleep(2)
                    ui.navigate.reload()
                
                ui.button('⏹️ Stop Recording', on_click=stop_and_process) \
                  .props('color=negative size=lg')
                
            elif not state.audio_recording and not state.tts_playing and state.waiting_for_answer:
                # Start recording button
                def start_recording():
                    start_audio_recording()
                    ui.navigate.reload()
                
                ui.button('🎤 Start Recording', on_click=start_recording) \
                  .props('color=primary size=lg')
            
            elif state.current_answer:
                # Submit answer button
                def submit_answer():
                    print(f"🔄 Submit button clicked - Answer: {state.current_answer[:50]}...")
                    if state.current_answer and state.current_answer.strip():
                        # Process in thread to avoid blocking UI
                        def process_and_reload():
                            process_answer(state.current_answer, state.current_audio_path)
                            # Small delay before reload to ensure processing completes
                            time.sleep(1)
                            ui.navigate.reload()
                        
                        threading.Thread(target=process_and_reload, daemon=True).start()
                        ui.notify('Processing your answer...', type='info')
                    else:
                        ui.notify('No answer to submit', type='warning')
                
                ui.button('✅ Submit Answer', on_click=submit_answer) \
                .props('color=positive size=lg')
                
                # Re-record button
                def rerecord():
                    state.current_answer = None
                    state.current_audio_path = None
                    start_audio_recording()
                    ui.navigate.reload()
                
                ui.button('🔄 Re-record', on_click=rerecord) \
                  .props('color=secondary')
        
        if state.question_num >= MIN_QUESTIONS:
            ui.separator().classes('my-4')
            ui.button('🏁 End Interview', 
                     on_click=lambda: (stop_video_recording(), complete_interview(), ui.navigate.to('/results'))) \
              .props('color=secondary')


def create_results_page():
    """Results page with XAI visualizations"""
    create_header()
    
    if not state.report:
        ui.label('No report available').classes('text-xl')
        ui.button('← Back', on_click=lambda: ui.navigate.to('/')).props('color=primary')
        return
    
    report = state.report
    score = report['scores']['overall_score']
    
    with ui.column().classes('w-full items-center p-4'):
        ui.label('📊 Interview Results').classes('text-4xl font-bold mb-8')
        
        # Overall score card
        with ui.card().classes('w-full max-w-6xl p-8 mb-6'):
            with ui.row().classes('w-full items-center gap-8'):
                with ui.column().classes('items-center'):
                    ui.label('Overall Score').classes('text-xl text-gray-600')
                    score_class = 'text-green-600' if score >= 8 else 'text-blue-600' if score >= 6.5 else 'text-orange-600' if score >= 5 else 'text-red-600'
                    ui.label(f'{score:.1f}/10').classes(f'text-6xl font-bold {score_class}')
                
                ui.separator().props('vertical')
                
                with ui.column().classes('flex-1'):
                    ui.label(report['recommendation']).classes('text-2xl font-bold mb-2')
                    ui.label(report['summary']).classes('text-lg text-gray-700')
        
        # Component scores
        with ui.card().classes('w-full max-w-6xl p-6 mb-6'):
            ui.label('📈 Component Scores').classes('text-2xl font-bold mb-4')
            
            scores_data = report['scores']
            components = [
                ('Motivation', scores_data['motivation']),
                ('Experience', scores_data['experience_fit']),
                ('Confidence', scores_data['confidence']),
                ('Logistics', scores_data['logistics']),
                ('Red Flags', scores_data['red_flags_score'])
            ]
            
            with ui.row().classes('w-full gap-4'):
                for name, val in components:
                    with ui.card().classes('flex-1 p-4 text-center'):
                        ui.label(name).classes('text-sm text-gray-600')
                        color = 'text-green-600' if val >= 7 else 'text-blue-600' if val >= 5 else 'text-orange-600'
                        ui.label(f'{val:.1f}').classes(f'text-3xl font-bold {color}')
                        ui.linear_progress(value=val/10).classes('mt-2')
        
        # XAI Explanations
        if 'xai_explanations' in report and report['xai_explanations']:
            with ui.card().classes('w-full max-w-6xl p-6 mb-6'):
                ui.label('🔍 XAI Analysis (Explainable AI)').classes('text-2xl font-bold mb-4')
                
                xai = report['xai_explanations']
                
                # Score breakdown
                if 'score' in xai and xai['score']:
                    with ui.expansion('Score Breakdown & Contributions', icon='insights').classes('w-full'):
                        score_xai = xai['score']
                        ui.label(score_xai.get('explanation', '')).classes('mb-4')
                        
                        contrib = score_xai.get('contributions', {})
                        if contrib:
                            for k, v in contrib.items():
                                with ui.row().classes('w-full items-center gap-4 mb-2'):
                                    ui.label(k.replace('_', ' ').title()).classes('w-40 font-semibold')
                                    ui.linear_progress(value=v['contribution']/10).classes('flex-1')
                                    ui.label(f"{v['contribution']:.2f}").classes('w-16 text-right')
                                    ui.label(f"({v['score']:.1f}/10)").classes('w-20 text-gray-600')
                
                # Audio emotion
                if 'audio_emotion' in xai and xai['audio_emotion']:
                    with ui.expansion('🎤 Audio Emotion Analysis', icon='mic').classes('w-full'):
                        audio_xai = xai['audio_emotion']
                        ui.label(f"Method: {audio_xai.get('method', 'N/A')}").classes('font-semibold')
                        ui.label(f"Dominant: {audio_xai.get('top_emotion', 'N/A').title()}").classes('text-lg')
                        ui.label(f"Confidence: {audio_xai.get('confidence', 0):.1%}").classes('text-lg')
                        ui.label(audio_xai.get('explanation', '')).classes('mt-2')
                        
                        if 'top_3_features' in audio_xai:
                            ui.label('Top Contributing Features:').classes('font-semibold mt-4')
                            for i, feat in enumerate(audio_xai['top_3_features'], 1):
                                feat_name = feat.get('feature', 'Unknown')
                                if 'shap_value' in feat:
                                    ui.label(f"{i}. {feat_name}: SHAP value = {feat['shap_value']:.3f} ({feat['impact']})")
                                elif 'lime_importance' in feat:
                                    ui.label(f"{i}. {feat_name}: LIME importance = {feat['lime_importance']:.3f}")
                                elif 'importance' in feat:
                                    ui.label(f"{i}. {feat_name}: Importance = {feat['importance']:.3f}")
                
                # Video emotion
                if 'video_emotion' in xai and xai['video_emotion']:
                    with ui.expansion('📹 Video Emotion Analysis', icon='videocam').classes('w-full'):
                        video_xai = xai['video_emotion']
                        ui.label(f"Dominant: {video_xai.get('dominant_emotion', 'N/A').title()}").classes('text-lg')
                        ui.label(f"Consistency: {video_xai.get('consistency', 0):.1%}").classes('text-lg')
                        ui.label(f"Trajectory: {video_xai.get('trajectory', 'N/A').title()}").classes('text-lg')
                        ui.label(video_xai.get('explanation', '')).classes('mt-2')
                        
                        if 'emotion_distribution' in video_xai:
                            ui.label('Emotion Distribution:').classes('font-semibold mt-4')
                            for emotion, score in sorted(video_xai['emotion_distribution'].items(), 
                                                        key=lambda x: x[1], reverse=True):
                                with ui.row().classes('w-full items-center gap-2'):
                                    ui.label(emotion.title()).classes('w-32')
                                    ui.linear_progress(value=score).classes('flex-1')
                                    ui.label(f'{score:.1%}').classes('w-16 text-right')
        
        # Feedback
        if 'feedback' in report and report['feedback']:
            with ui.card().classes('w-full max-w-6xl p-6 mb-6'):
                ui.label('💡 Detailed Feedback').classes('text-2xl font-bold mb-4')
                
                feedback = report['feedback']
                
                with ui.row().classes('w-full gap-4'):
                    # Strengths
                    with ui.column().classes('flex-1'):
                        ui.label('✅ Strengths').classes('text-xl font-semibold mb-2')
                        for strength in feedback.get('strengths', []):
                            with ui.card().classes('p-3 mb-2 bg-green-50'):
                                ui.label(strength).classes('text-green-800')
                    
                    # Improvements
                    with ui.column().classes('flex-1'):
                        ui.label('📈 Areas for Improvement').classes('text-xl font-semibold mb-2')
                        for area in feedback.get('areas_for_improvement', []):
                            with ui.card().classes('p-3 mb-2 bg-orange-50'):
                                ui.label(area).classes('text-orange-800')
                
                # Specific recommendations
                if feedback.get('specific_recommendations'):
                    ui.separator().classes('my-4')
                    ui.label('🎯 Specific Recommendations').classes('text-xl font-semibold mb-2')
                    for rec in feedback['specific_recommendations']:
                        with ui.card().classes('p-3 mb-2 bg-blue-50'):
                            ui.label(rec).classes('text-blue-800')
                
                # XAI methods used
                if feedback.get('xai_methods_used'):
                    with ui.expansion('🔬 XAI Methods Used', icon='science').classes('w-full mt-4'):
                        ui.label('This analysis used the following explainable AI techniques:')
                        for method in feedback['xai_methods_used']:
                            ui.label(f'• {method}')
        
        # Key Insights
        if 'key_insights' in report:
            with ui.card().classes('w-full max-w-6xl p-6 mb-6'):
                ui.label('💡 Key Insights').classes('text-2xl font-bold mb-4')
                
                insights = report['key_insights']
                
                with ui.row().classes('w-full gap-4 mb-4'):
                    with ui.card().classes('flex-1 p-4 text-center'):
                        ui.label('Availability').classes('text-sm text-gray-600')
                        ui.label(insights.get('availability', 'N/A')).classes('text-xl font-bold')
                    
                    with ui.card().classes('flex-1 p-4 text-center'):
                        ui.label('Motivation Level').classes('text-sm text-gray-600')
                        ui.label(insights.get('motivation_level', 'N/A').title()).classes('text-xl font-bold')
                    
                    with ui.card().classes('flex-1 p-4 text-center'):
                        ui.label('Concerns Detected').classes('text-sm text-gray-600')
                        concerns = insights.get('concerns', [])
                        ui.label(str(len(concerns))).classes('text-xl font-bold')
                
                if insights.get('skills_mentioned'):
                    ui.label('Skills Mentioned:').classes('font-semibold')
                    ui.label(', '.join(insights['skills_mentioned'][:10]))
                
                if concerns:
                    with ui.expansion(f'⚠️ View {len(concerns)} Concern(s)', icon='warning').classes('w-full'):
                        for concern in concerns:
                            with ui.card().classes('p-3 mb-2 bg-red-50'):
                                ui.label(concern).classes('text-red-800')
        
        # Transcript
        with ui.card().classes('w-full max-w-6xl p-6 mb-6'):
            ui.label('📝 Interview Transcript').classes('text-2xl font-bold mb-4')
            
            if 'transcript' in report:
                for i, qa in enumerate(report['transcript'], 1):
                    with ui.expansion(f"Q{i}: {qa['question'][:60]}...", icon='chat').classes('w-full'):
                        ui.label('Question:').classes('font-semibold')
                        ui.label(qa['question']).classes('mb-2')
                        ui.label('Answer:').classes('font-semibold')
                        ui.label(qa['answer']).classes('text-gray-700')
        
        # Actions
        with ui.row().classes('gap-4'):
            # Download report
            report_json = json.dumps(report, indent=2)
            ui.button(
                '📥 Download Report',
                on_click=lambda: ui.download(
                    report_json.encode(),
                    f"interview_{report['metadata']['candidate_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            ).props('color=primary')
            
            # New interview
            ui.button(
                '🔄 New Interview',
                on_click=lambda: (reset_interview(), ui.navigate.to('/'))
            ).props('color=secondary')
    
    create_footer()


# =============================================================================
# EVENT HANDLERS
# =============================================================================

def on_start_interview(name: str, job_title: str, skills: str, mode: str):
    """Handle interview start"""
    if not name.strip():
        ui.notify('Please enter candidate name', type='warning')
        return
    
    if not job_title.strip():
        ui.notify('Please enter job title', type='warning')
        return
    
    # Show loading state
    ui.notify('🔄 Starting interview... Please wait', type='info')
    
    start_interview(name, job_title, skills, mode)
    
    # Navigate to interview page FIRST
    ui.navigate.to('/interview')
    
    # Start TTS and recording for voice/video modes AFTER navigation
    if mode in ['voice', 'video']:
        # Wait longer for page to fully load before starting TTS
        def delayed_tts():
            time.sleep(3)  # Increased wait time
            if state.current_question and not state.tts_playing:
                print("🎯 Starting TTS after page load...")
                speak_question(state.current_question)
                
                # Wait for TTS to finish, then start recording
                def start_after_tts():
                    # Wait for TTS to actually start
                    time.sleep(1)
                    while state.tts_playing:
                        time.sleep(0.1)
                    print("🎯 TTS finished, starting recording...")
                    start_audio_recording()
                    state.needs_auto_record = False
                
                threading.Thread(target=start_after_tts, daemon=True).start()
        
        threading.Thread(target=delayed_tts, daemon=True).start()

def show_loading(message: str = "Loading..."):
    """Show loading indicator"""
    with ui.dialog() as dialog, ui.card():
        ui.spinner(size='lg')
        ui.label(message)
    dialog.open()
    return dialog

# =============================================================================
# AUTO-UPDATE TIMERS
# =============================================================================

async def check_audio_status():
    """Check audio recording status and update UI"""
    if state.audio_recording and state.audio_state.get('recording_complete', False):
        print("🔄 Audio recording complete, updating state...")
        state.current_answer = state.audio_state.get('answer')
        state.current_audio_path = state.audio_state.get('audio_path')
        state.audio_recording = False
        
        # Only auto-process if we have a valid answer
        if state.current_answer and state.current_answer.strip():
            print(f"🔄 Auto-processing answer: {state.current_answer[:50]}...")
            # Use threading to avoid blocking the async function
            def process_in_thread():
                process_answer(state.current_answer, state.current_audio_path)
                # Schedule UI reload after processing
                ui.timer(1.0, lambda: ui.navigate.reload(), once=True)
            
            threading.Thread(target=process_in_thread, daemon=True).start()
        else:
            # Just reload UI if no answer
            ui.navigate.reload()


async def check_tts_status():
    """Check TTS status and auto-start recording"""
    if (state.needs_auto_record and 
        not state.tts_playing and 
        not state.audio_recording and
        state.current_answer is None):
        state.needs_auto_record = False
        start_audio_recording()
        ui.navigate.reload()


async def auto_speak_question():
    """Auto-speak question if needed"""
    if (state.needs_tts and 
        not state.tts_playing and
        state.mode in ['voice', 'video']):
        state.needs_tts = False
        speak_question(state.current_question)
        state.needs_auto_record = True


# =============================================================================
# ROUTES (FIXED VERSION)
# =============================================================================

@ui.page('/')
def index():
    """Welcome page"""
    if state.interview_started and not state.report:
        ui.navigate.to('/interview')
        return
    elif state.report:
        ui.navigate.to('/results')
        return
    
    create_welcome_page()

@ui.page('/interview')
def interview():
    """Interview page"""
    if not state.interview_started:
        ui.navigate.to('/')
        return
    
    if state.report:
        ui.navigate.to('/results')
        return
    
    create_interview_page()
    
    # Set up auto-update timers
    ui.timer(0.5, check_audio_status, active=True)
    ui.timer(0.5, check_tts_status, active=True)
    ui.timer(0.5, auto_speak_question, active=True)

@ui.page('/results')
def results():
    """Results page"""
    if not state.report:
        ui.navigate.to('/')
        return
    
    create_results_page()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Initialize and run the application"""
    ui.run(
        title='AI Interviewer',
        port=8080,
        reload=False,
        show=True
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()