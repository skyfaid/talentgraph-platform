# Voice Features Setup Guide

## 🎤 AI Voice Reading & Voice Input for Interviews

This guide explains how to set up voice features for the interview system.

## Features

1. **AI Voice Reading (TTS)**: Questions are read aloud with high-quality AI voices
2. **Voice Input (STT)**: Answer questions by speaking instead of typing
3. **Emotion Detection**: (Future) Analyze emotions from voice

## ✅ 100% FREE - No API Keys Required!

All voice features use completely free services:
- **TTS**: edge-tts (Microsoft Edge TTS) - FREE, no API key
- **STT**: OpenAI Whisper - FREE, runs locally
- **No costs, no limits!**

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `edge-tts` - FREE TTS (Microsoft Edge voices, no API key needed)
- `openai-whisper` - FREE speech-to-text (runs locally)
- `audio-recorder-streamlit` - Microphone input in Streamlit

### 2. No API Keys Needed!

Everything is free and works out of the box. No setup required!

### 3. First Run Setup

On first run, Whisper will download the model (~150MB). This happens automatically.

## How It Works

### Voice Reading (TTS)

1. When a question appears, the system automatically generates audio
2. An audio player appears below the question
3. Click play to hear the question read aloud
4. Audio is cached (generated once per question)

**Voice Options:**
- **edge-tts**: Professional Microsoft voices (completely FREE)
- Multiple voice options: Aria, Jenny, Guy, etc.
- High-quality, natural-sounding speech
- No API key, no limits, no costs!

### Voice Input (STT)

1. Go to the "🎤 Voice Answer" tab
2. Click the microphone button to start recording
3. Speak your answer
4. Click stop when finished
5. Your speech is automatically converted to text
6. The transcribed text appears in the answer field

**Speech Recognition:**
- Uses OpenAI Whisper (offline, accurate)
- Supports multiple languages
- Automatic punctuation and formatting

## Future: Emotion Detection

The system is structured to support emotion detection from voice. When implemented, it will:

1. Analyze voice tone and patterns
2. Detect emotions: happy, sad, neutral, confident, nervous, etc.
3. Provide insights on candidate's emotional state during interview
4. Help assess communication skills and confidence

**Planned Implementation:**
- Use audio features (pitch, tone, speed)
- ML models for emotion classification
- Integration with answer evaluation

## Troubleshooting

### "TTS not available"
- Check that API keys are set correctly
- Verify ElevenLabs/OpenAI packages are installed
- Check internet connection (for API calls)

### "Voice input not available"
- Make sure `streamlit-audio-recorder` is installed
- Grant microphone permissions in browser
- Check browser compatibility (Chrome/Edge recommended)

### "Whisper model loading error"
- First run downloads model automatically
- Check internet connection
- Model size: ~150MB (base model)

### Audio quality issues
- ElevenLabs provides best quality
- Try different voice options
- Check your internet connection for API calls

## Voice Options

### ElevenLabs Voices
- Professional voices (Rachel, Adam, etc.)
- Celebrity voices (if available on your plan)
- Custom voices (premium feature)

### OpenAI Voices
- `alloy` - Neutral, balanced
- `echo` - Warm, friendly
- `fable` - Expressive
- `onyx` - Deep, authoritative
- `nova` - Bright, energetic
- `shimmer` - Soft, gentle

## Cost: $0.00 - Completely FREE!

- **edge-tts (TTS)**: FREE - No limits, no API key needed
- **Whisper (STT)**: FREE - Runs locally, open-source
- **Total cost**: $0.00

## Example Usage

1. **Start Interview** → Question appears
2. **Audio Player** → Click play to hear question (auto-generated)
3. **Voice Answer Tab** → Record your answer
4. **Auto-Transcribe** → Text appears automatically (FREE)
5. **Submit** → Answer is evaluated

## Notes

- Audio is cached per question (generated once)
- Voice input works offline (Whisper runs locally)
- Emotion detection is planned for future release
- All voice features are optional (system works without them)
- **No API keys needed** - Everything works out of the box!

