"""
Audio Emotion Recognition using CNN on EmoDB + RAVDESS
"""
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from utils.config import EMODB_PATH, RAVDESS_PATH, EMOTION_LABELS, SAMPLE_RATE, MODELS_DIR


class EmotionCNN(nn.Module):
    """CNN for emotion classification from audio MFCCs"""
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Adjusted dimensions for 40x100 input MFCC features
        self.fc1 = nn.Linear(128 * 5 * 12, 256)  # After 3 conv+pool layers: 128 * 5 * 12
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AudioEmotionClassifier:
    def __init__(self):
        """Initialize emotion classifier"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = MODELS_DIR / "emotion_cnn.pth"
        self.encoder_path = MODELS_DIR / "label_encoder.pkl"
        
    def extract_mfcc_features(self, audio_path: str, n_mfcc=40, max_len=100) -> np.ndarray:
        """Extract MFCC features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=3)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Pad or truncate to fixed length
            if mfccs.shape[1] < max_len:
                pad_width = max_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_len]
            
            return mfccs
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros((n_mfcc, max_len))
    
    def load_dataset(self) -> tuple:
        """Load EmoDB and RAVDESS datasets"""
        X, y = [], []
        
        # Load EmoDB (German)
        if EMODB_PATH.exists():
            print("Loading EmoDB...")
            emodb_map = {'W': 'angry', 'L': 'sad', 'E': 'disgust', 'A': 'fearful', 
                        'F': 'happy', 'T': 'sad', 'N': 'neutral'}
            
            for audio_file in EMODB_PATH.glob("*.wav"):
                emotion_code = audio_file.stem[5]
                if emotion_code in emodb_map:
                    emotion = emodb_map[emotion_code]
                    mfcc = self.extract_mfcc_features(str(audio_file))
                    X.append(mfcc)
                    y.append(emotion)
        
        # Load RAVDESS (English)
        if RAVDESS_PATH.exists():
            print("Loading RAVDESS...")
            ravdess_map = {
                '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
            }
            
            for audio_file in RAVDESS_PATH.rglob("*.wav"):
                parts = audio_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in ravdess_map:
                        emotion = ravdess_map[emotion_code]
                        mfcc = self.extract_mfcc_features(str(audio_file))
                        X.append(mfcc)
                        y.append(emotion)
        
        if len(X) == 0:
            raise ValueError("No audio files found! Check dataset paths.")
        
        return np.array(X), np.array(y)
    
    def train(self, epochs=30, batch_size=32):
        """Train the emotion classifier"""
        print("Loading datasets...")
        X, y = self.load_dataset()
        
        print(f"Dataset size: {len(X)} samples")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = EmotionCNN(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print("Training model...")
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                # Evaluate
                self.model.eval()
                with torch.no_grad():
                    X_test_device = X_test.to(self.device)
                    y_test_device = y_test.to(self.device)
                    outputs = self.model(X_test_device)
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_test_device).sum().item() / len(y_test)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        self.save_model()
        print(f"Model saved to {self.model_path}")
    
    def save_model(self):
        """Save model and label encoder"""
        torch.save(self.model.state_dict(), self.model_path)
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError("Model not found! Train the model first.")
        
        with open(self.encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        num_classes = len(self.label_encoder.classes_)
        self.model = EmotionCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, audio_path: str) -> Dict[str, float]:
        """Predict emotion from audio file"""
        if self.model is None:
            self.load_model()
        
        mfcc = self.extract_mfcc_features(audio_path)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(mfcc_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        emotions = self.label_encoder.classes_
        emotion_scores = {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
        
        # Calculate confidence score (for interview scoring)
        confidence_emotions = ['happy', 'neutral', 'calm']
        nervous_emotions = ['fearful', 'sad']
        
        confidence_score = sum(emotion_scores.get(e, 0) for e in confidence_emotions)
        nervous_score = sum(emotion_scores.get(e, 0) for e in nervous_emotions)
        
        emotion_scores['confidence_score'] = (confidence_score - nervous_score) * 10
        emotion_scores['confidence_score'] = min(10, max(1, emotion_scores['confidence_score']))
        
        return emotion_scores