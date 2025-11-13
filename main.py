"""
AI Interviewer - Main Entry Point
Conducts automated screening interviews with voice/text/video analysis
"""
import argparse
import time
import threading
from pathlib import Path
from typing import Dict, List

from core.audio_processor import AudioProcessor
from core.video_processor import VideoProcessor
from core.question_generator import QuestionGenerator
from core.answer_evaluator import AnswerEvaluator
from models.text_analysis import TextAnalyzer
from models.audio_emotion_model import AudioEmotionClassifier
from scoring.scoring_system import ScoringSystem
from utils.config import MIN_QUESTIONS, MAX_QUESTIONS, TARGET_DURATION_MINUTES


class AIInterviewer:
    def __init__(self, mode='text'):
        """
        Initialize AI Interviewer
        
        Args:
            mode: 'text', 'voice', or 'video'
        """
        self.mode = mode
        
        print("Initializing AI Interviewer...")
        self.audio_processor = AudioProcessor() if mode in ['voice', 'video'] else None
        self.video_processor = VideoProcessor() if mode == 'video' else None
        self.question_generator = QuestionGenerator()
        self.answer_evaluator = AnswerEvaluator()
        self.text_analyzer = TextAnalyzer()
        self.audio_emotion = AudioEmotionClassifier() if mode in ['voice', 'video'] else None
        self.scoring_system = ScoringSystem()
        
        # Video preview thread control
        self.video_preview_active = False
        self.video_preview_thread = None
        
        if mode in ['voice', 'video'] and self.audio_emotion:
            try:
                self.audio_emotion.load_model()
            except FileNotFoundError:
                print("Audio emotion model not found. Run training first with --train flag.")
        
        self.interview_data = {
            'qa_pairs': [],
            'answer_evaluations': [],
            'red_flags': [],
            'extracted_entities': {'skills': [], 'organizations': [], 'dates': []},
            'text_sentiment': {},
            'audio_emotion': {},
            'video_emotion': {},
            'availability': {},
            'candidate_name': None,
            'skill_match': {}
        }
    
    def _continuous_video_preview(self):
        """Show continuous video preview during interview"""
        import cv2
        
        print("📹 Video preview window opened (close window or press 'q' to hide)")
        
        while self.video_preview_active and self.video_processor:
            frame = self.video_processor.get_current_frame()
            
            if frame is not None:
                # Add recording indicator
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show latest emotion if available
                if self.video_processor.emotion_history:
                    latest = self.video_processor.emotion_history[-1]['emotions']
                    dominant = max(latest.items(), key=lambda x: x[1])
                    cv2.putText(frame, f"{dominant[0]}: {dominant[1]:.2f}", 
                               (10, frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('AI Interview - Video Mode', frame)
            
            # Check for window close or 'q' key
            key = cv2.waitKey(30)
            if key & 0xFF == ord('q'):
                break
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty('AI Interview - Video Mode', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break
            
            time.sleep(0.03)  # ~30 FPS
        
        cv2.destroyAllWindows()
        print("📹 Video preview closed")
    
    def conduct_interview(
        self, 
        candidate_name: str = "Candidate",
        job_context: Dict = None
    ) -> Dict:
        """
        Main interview loop
        
        Args:
            candidate_name: Name of candidate
            job_context: Job details dict
            
        Returns:
            Complete interview report
        """
        self.interview_data['candidate_name'] = candidate_name
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting Interview with {candidate_name}")
        print(f"Mode: {self.mode.upper()}")
        print(f"{'='*60}\n")
        
        # Start video recording at the beginning (if video mode)
        if self.mode == 'video' and self.video_processor:
            output_path = str(self.video_processor.temp_dir / f"interview_{candidate_name}_{int(time.time())}.mp4")
            self.video_processor.start_recording(output_path, fps=15)
            time.sleep(1.0)  # Wait for camera to initialize
            
            # Start continuous preview in separate thread
            self.video_preview_active = True
            self.video_preview_thread = threading.Thread(
                target=self._continuous_video_preview,
                daemon=True
            )
            self.video_preview_thread.start()
            print("✓ Video recording started\n")
        
        question_count = 0
        
        try:
            while question_count < MAX_QUESTIONS:
                # Generate question
                if question_count == 0:
                    question = self.question_generator.generate_question(job_context=job_context)
                else:
                    prev_answer = self.interview_data['qa_pairs'][-1]['answer']
                    question = self.question_generator.generate_question(prev_answer, job_context)
                
                print(f"\n[Q{question_count + 1}] AI: {question}")
                
                # Speak question in voice/video mode (with proper engine reset)
                if self.mode in ['voice', 'video'] and self.audio_processor:
                    print("🔊 Speaking question...")
                    try:
                        self.audio_processor.speak(question)
                    except Exception as e:
                        print(f"⚠️ TTS error: {e}. Question displayed as text.")
                
                # Get candidate answer
                if self.mode in ['voice', 'video']:
                    answer, audio_path = self._get_voice_answer()
                else:
                    answer = self._get_text_answer()
                    audio_path = None
                
                if not answer or answer.lower() in ['quit', 'exit', 'end']:
                    print("\nInterview ended by candidate.")
                    break
                
                print(f"[A{question_count + 1}] Candidate: {answer}")
                
                # Analyze answer
                self._analyze_answer(question, answer, audio_path, job_context)
                
                question_count += 1
                
                # Check if minimum questions met and key topics covered
                coverage = self.question_generator.get_coverage_status()
                if question_count >= MIN_QUESTIONS and sum(coverage.values()) >= 3:
                    print("\n[INFO] Key topics covered. Interview can conclude.")
                    user_input = input("Continue? (y/n): ").strip().lower()
                    if user_input != 'y':
                        break
        
        finally:
            # Stop video recording and preview
            if self.mode == 'video' and self.video_processor:
                self.video_preview_active = False
                if self.video_preview_thread:
                    self.video_preview_thread.join(timeout=2.0)
                self.video_processor.stop_recording()
                print("✓ Video recording stopped")
        
        duration = (time.time() - start_time) / 60
        self.interview_data['duration'] = round(duration, 2)
        
        # Calculate scores and generate report
        print("\n" + "="*60)
        print("Generating Report...")
        print("="*60)
        
        scores = self.scoring_system.calculate_final_score(self.interview_data)
        report = self.scoring_system.generate_report(self.interview_data, scores)
        
        # Save report
        report_path = self.scoring_system.save_report(report)
        
        # Display summary
        self._display_summary(report)
        
        return report
    
    def _get_text_answer(self) -> str:
        """Get text answer from candidate"""
        return input("[Your answer]: ").strip()
    
    def _get_voice_answer(self) -> tuple:
        """Get voice answer from candidate"""
        # In video mode, recording is already running continuously
        # We just need to capture the audio for this specific answer
        
        print("🎤 Recording your answer... (Press ENTER when done)")
        
        # Record audio
        audio_path = self.audio_processor.record_audio(duration=None, max_duration=60)
        
        if not audio_path:
            print("⚠️  No audio recorded. Please type your answer:")
            answer = input("[Your answer]: ").strip()
            return answer, None
        
        # Transcribe
        print("🔄 Transcribing audio...")
        answer = self.audio_processor.transcribe_audio(audio_path)
        
        if not answer:
            print("⚠️  Transcription failed. Please type your answer:")
            answer = input("[Your answer]: ").strip()
            return answer, None
        
        print(f"📝 Transcribed: {answer}")
        return answer, audio_path
    
    def _analyze_answer(
        self, 
        question: str, 
        answer: str, 
        audio_path: str = None,
        job_context: Dict = None
    ):
        """Analyze candidate answer with all models"""
        
        # 1. Store Q&A pair
        qa_pair = {'question': question, 'answer': answer}
        self.interview_data['qa_pairs'].append(qa_pair)
        
        # 2. Evaluate answer quality with Mistral
        print("  [Analyzing answer quality...]")
        evaluation = self.answer_evaluator.evaluate_answer(question, answer, job_context)
        self.interview_data['answer_evaluations'].append(evaluation)
        
        # 3. Text sentiment analysis
        print("  [Analyzing sentiment...]")
        sentiment = self.text_analyzer.analyze_sentiment(answer)
        self.interview_data['text_sentiment'] = sentiment
        
        # 4. Extract entities
        print("  [Extracting information...]")
        entities = self.text_analyzer.extract_entities(answer)
        for key, vals in entities.items():
            if key not in self.interview_data['extracted_entities']:
                self.interview_data['extracted_entities'][key] = []
            if vals:
                try:
                    self.interview_data['extracted_entities'][key].extend(vals)
                except TypeError:
                    self.interview_data['extracted_entities'][key].append(vals)
        
        # 5. Extract availability
        availability = self.text_analyzer.extract_availability(answer)
        if availability['notice_period'] or availability['available_immediately']:
            self.interview_data['availability'] = availability
        
        # 6. Detect red flags
        red_flags = self.text_analyzer.detect_red_flags(answer)
        self.interview_data['red_flags'].extend(red_flags)
        
        # 7. Audio emotion analysis (if voice or video mode)
        if self.mode in ['voice', 'video'] and audio_path and self.audio_emotion:
            print("  [Analyzing voice emotion...]")
            try:
                audio_emotion = self.audio_emotion.predict(audio_path)
                self.interview_data['audio_emotion'] = audio_emotion
            except Exception as e:
                print(f"  [Warning] Audio emotion analysis failed: {e}")
        
        # 8. Video emotion analysis (if video mode)
        # Note: We don't analyze per-answer, we'll aggregate at the end
        if self.mode == 'video' and self.video_processor:
            print("  [Capturing facial emotions...]")
            try:
                # Get current emotion snapshot
                if self.video_processor.emotion_history:
                    latest = self.video_processor.emotion_history[-1]['emotions']
                    dominant = max(latest.items(), key=lambda x: x[1])
                    print(f"    Current expression: {dominant[0]} ({dominant[1]:.2f})")
            except Exception as e:
                print(f"  [Warning] Video emotion check failed: {e}")
        
        print("  [Analysis complete]\n")
    
    def _display_summary(self, report: Dict):
        """Display interview summary"""
        print("\n" + "="*60)
        print("INTERVIEW SUMMARY")
        print("="*60)
        
        scores = report['scores']
        print(f"\nOverall Score: {scores['overall_score']}/10")
        print(f"Recommendation: {report['recommendation']}")
        
        print("\n--- Component Scores ---")
        print(f"  Motivation:     {scores['motivation']:.1f}/10")
        print(f"  Experience Fit: {scores['experience_fit']:.1f}/10")
        print(f"  Confidence:     {scores['confidence']:.1f}/10")
        print(f"  Logistics:      {scores['logistics']:.1f}/10")
        
        print("\n--- Key Insights ---")
        insights = report['key_insights']
        print(f"  Availability: {insights['availability']}")
        print(f"  Motivation: {insights['motivation_level'].title()}")
        
        if insights['skills_mentioned']:
            print(f"  Skills: {', '.join(insights['skills_mentioned'][:5])}")
        
        if insights['concerns']:
            print(f"  ⚠️  Concerns: {len(insights['concerns'])} red flags detected")
        
        # Show video emotion summary if available
        if self.mode == 'video' and self.video_processor:
            print("\n--- Video Emotion Analysis ---")
            dominant = self.video_processor.get_dominant_emotions(top_n=3)
            for i, (emotion, score) in enumerate(dominant, 1):
                print(f"  {i}. {emotion.capitalize()}: {score:.2f}")
            
            # Get final aggregated emotions for report
            video_emotions = self.video_processor.get_aggregated_emotions()
            self.interview_data['video_emotion'] = video_emotions
        
        print("\n" + "="*60)


def train_emotion_model():
    """Train audio emotion classifier"""
    print("Training Audio Emotion Classifier...")
    classifier = AudioEmotionClassifier()
    classifier.train(epochs=30, batch_size=32)
    print("Training complete!")


def test_video_mode():
    """Test video recording and emotion detection"""
    print("\n" + "="*60)
    print("🟢 Testing Video Mode: Real-Time Emotion Detection")
    print("="*60 + "\n")
    
    from core.video_processor import VideoProcessor
    
    processor = VideoProcessor()
    duration = 10  # seconds
    
    print(f"Starting {duration} second test recording...")
    print("Make different facial expressions to test emotion detection!")
    print("-" * 60)
    
    try:
        # Start recording
        processor.start_recording()
        
        # Show preview while recording
        processor.show_preview_window(duration=duration)
        
        # Stop recording
        processor.stop_recording()
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        # Show aggregated emotions
        print("\nAggregated Emotion Scores:")
        results = processor.get_aggregated_emotions()
        for emotion, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {emotion.capitalize():12s} [{score:.3f}] {bar}")
        
        # Show dominant emotions
        print("\nTop 3 Dominant Emotions:")
        for i, (emotion, score) in enumerate(processor.get_dominant_emotions(3), 1):
            print(f"  {i}. {emotion.capitalize()}: {score:.3f}")
        
        # Show timeline
        print(f"\nEmotion Timeline ({len(processor.get_emotion_timeline())} snapshots):")
        for entry in processor.get_emotion_timeline()[:5]:  # Show first 5
            ts = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            top_emotion = max(entry['emotions'].items(), key=lambda x: x[1])
            print(f"  [{ts}] {top_emotion[0]}: {top_emotion[1]:.2f}")
        
        if len(processor.get_emotion_timeline()) > 5:
            print(f"  ... and {len(processor.get_emotion_timeline()) - 5} more snapshots")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        processor.stop_recording()
    except Exception as e:
        print(f"\n⚠️ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.cleanup_temp_files()
        print("\n✓ Test complete. Temp files cleaned up.")


def main():
    parser = argparse.ArgumentParser(description="AI Interviewer System")
    parser.add_argument('--mode', choices=['text', 'voice', 'video', 'interactive'], 
                       default='interactive',
                       help='Interview mode')
    parser.add_argument('--name', type=str, default=None,
                       help='Candidate name')
    parser.add_argument('--job-title', type=str, default=None,
                       help='Job title')
    parser.add_argument('--train', action='store_true',
                       help='Train audio emotion model')
    parser.add_argument('--test-video', action='store_true',
                       help='Test video recording and emotion detection')
    
    args = parser.parse_args()
    
    # Handle special flags
    if args.train:
        train_emotion_model()
        return
    
    if args.test_video:
        test_video_mode()
        return
    
    # Interactive mode - show menu
    if args.mode == 'interactive':
        print("\n" + "="*60)
        print("🤖 AI INTERVIEWER SYSTEM")
        print("="*60)
        print("\nSelect Interview Mode:")
        print("  1. Text Interview (Type answers)")
        print("  2. Voice Interview (Speak answers)")
        print("  3. Video Interview (Speak + Face analysis)")
        print("  4. Train Audio Emotion Model")
        print("  5. Test Video Mode")
        print("  6. Exit")
        print("\n" + "="*60)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            mode = 'text'
        elif choice == '2':
            mode = 'voice'
        elif choice == '3':
            mode = 'video'
        elif choice == '4':
            train_emotion_model()
            return
        elif choice == '5':
            test_video_mode()
            return
        elif choice == '6':
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Defaulting to text mode.")
            mode = 'text'
    else:
        mode = args.mode
    
    # Get interview parameters
    if args.name:
        candidate_name = args.name
    else:
        candidate_name = input("\nEnter candidate name: ").strip() or "Candidate"
    
    job_title = args.job_title
    if not args.job_title and args.mode == 'interactive':
        job_title = input("Enter job title (default: Software Engineer): ").strip() or "Software Engineer"
    
    # Job context
    job_context = {
        'title': job_title,
        'skills': ['python', 'machine learning', 'backend development']
    }
    
    # Run interview
    print(f"\n🚀 Starting {mode.upper()} mode interview...")
    
    try:
        interviewer = AIInterviewer(mode=mode)
        report = interviewer.conduct_interview(
            candidate_name=candidate_name,
            job_context=job_context
        )
        
        print("\n✅ Interview completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interview interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during interview: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if mode in ['voice', 'video'] and 'interviewer' in locals() and hasattr(interviewer, 'audio_processor'):
                interviewer.audio_processor.cleanup_temp_files()
            if mode == 'video' and 'interviewer' in locals() and hasattr(interviewer, 'video_processor'):
                interviewer.video_processor.cleanup_temp_files()
        except:
            pass


if __name__ == "__main__":
    main()