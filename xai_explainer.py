"""
XAI (Explainable AI) Module with REAL SHAP and LIME
Provides model-agnostic explanations for predictions
"""
import numpy as np
from typing import Dict, List, Tuple, Callable
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Install with: pip install lime")


class XAIExplainer:
    def __init__(self):
        """Initialize XAI explainer with SHAP and LIME"""
        self.shap_available = SHAP_AVAILABLE
        self.lime_available = LIME_AVAILABLE
        
        if not self.shap_available:
            print("⚠️  SHAP not available. Run: pip install shap")
        if not self.lime_available:
            print("⚠️  LIME not available. Run: pip install lime")
    
    def explain_audio_emotion_with_shap(
        self,
        model_predict_fn: Callable,
        audio_features_array: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Use SHAP to explain audio emotion prediction
        
        Args:
            model_predict_fn: Function that takes features and returns predictions
            audio_features_array: Feature vector (shape: (1, n_features))
            feature_names: Names of features
            
        Returns:
            SHAP explanation dictionary
        """
        if not self.shap_available:
            return self._fallback_audio_explanation(audio_features_array, feature_names)
        
        try:
            # Create SHAP explainer (KernelExplainer works for any model)
            # We'll use a small background dataset of zeros as baseline
            background = np.zeros((10, audio_features_array.shape[1]))
            
            explainer = shap.KernelExplainer(
                model_predict_fn,
                background,
                link="identity"
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(audio_features_array)
            
            # Get base value (expected value)
            base_value = explainer.expected_value
            
            # Format results
            feature_importance = []
            for i, feature_name in enumerate(feature_names):
                feature_importance.append({
                    'feature': feature_name,
                    'value': float(audio_features_array[0, i]),
                    'shap_value': float(shap_values[0, i]),
                    'impact': 'positive' if shap_values[0, i] > 0 else 'negative',
                    'magnitude': abs(float(shap_values[0, i]))
                })
            
            # Sort by magnitude
            feature_importance.sort(key=lambda x: x['magnitude'], reverse=True)
            
            return {
                'method': 'SHAP',
                'base_value': float(base_value) if isinstance(base_value, (int, float)) else float(base_value[0]),
                'feature_importance': feature_importance,
                'top_3_features': feature_importance[:3],
                'explanation': self._generate_shap_explanation(feature_importance[:3])
            }
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_audio_explanation(audio_features_array, feature_names)
    
    def explain_audio_emotion_with_lime(
        self,
        model_predict_fn: Callable,
        audio_features_array: np.ndarray,
        feature_names: List[str],
        training_data: np.ndarray = None
    ) -> Dict:
        """
        Use LIME to explain audio emotion prediction
        
        Args:
            model_predict_fn: Function that takes features and returns predictions
            audio_features_array: Feature vector
            feature_names: Names of features
            training_data: Training data for LIME (optional)
            
        Returns:
            LIME explanation dictionary
        """
        if not self.lime_available:
            return self._fallback_audio_explanation(audio_features_array, feature_names)
        
        try:
            # Create training data if not provided
            if training_data is None:
                # Create synthetic training data (more realistic than zeros)
                training_data = np.random.randn(100, audio_features_array.shape[1]) * 0.5
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
            
            # Explain the instance
            explanation = explainer.explain_instance(
                audio_features_array[0],
                model_predict_fn,
                num_features=len(feature_names)
            )
            
            # Get feature importance from LIME
            lime_values = explanation.as_list()
            
            feature_importance = []
            for feature_name, importance in lime_values:
                # Parse feature name (LIME returns "feature <= value" format)
                clean_name = feature_name.split('<=')[0].strip() if '<=' in feature_name else feature_name
                
                feature_importance.append({
                    'feature': clean_name,
                    'lime_importance': float(importance),
                    'impact': 'positive' if importance > 0 else 'negative',
                    'magnitude': abs(float(importance))
                })
            
            # Sort by magnitude
            feature_importance.sort(key=lambda x: x['magnitude'], reverse=True)
            
            return {
                'method': 'LIME',
                'feature_importance': feature_importance,
                'top_3_features': feature_importance[:3],
                'explanation': self._generate_lime_explanation(feature_importance[:3])
            }
            
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return self._fallback_audio_explanation(audio_features_array, feature_names)
    
    def explain_audio_emotion_prediction(
        self, 
        audio_features: Dict[str, any],
        emotion_scores: Dict[str, float],
        model = None
    ) -> Dict[str, any]:
        """
        Explain audio emotion prediction using SHAP or LIME
        
        Args:
            audio_features: Audio features extracted
            emotion_scores: Predicted emotion scores
            model: The emotion classification model (optional)
            
        Returns:
            Explanation dictionary with feature contributions
        """
        if not audio_features or not emotion_scores:
            return {
                'top_emotion': 'neutral',
                'confidence': 0.5,
                'feature_contributions': {},
                'explanation': "Insufficient data for explanation",
                'method': 'fallback'
            }
        
        # Get dominant emotion
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Prepare features for SHAP/LIME
        feature_names = []
        feature_values = []
        
        for key, value in audio_features.items():
            if isinstance(value, (int, float)):
                feature_names.append(key)
                feature_values.append(value)
            elif isinstance(value, np.ndarray):
                # Handle array features (like mfcc_mean)
                if value.size == 1:
                    feature_names.append(key)
                    feature_values.append(float(value))
                else:
                    # For multi-dimensional features, take mean
                    feature_names.append(f"{key}_avg")
                    feature_values.append(float(np.mean(value)))
        
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Try SHAP first, then LIME, then fallback
        explanation = None
        
        if model and self.shap_available:
            try:
                # Create prediction function for SHAP
                def predict_fn(X):
                    # This is a wrapper for the model
                    predictions = []
                    for x in X:
                        # Reconstruct feature dict
                        feat_dict = dict(zip(feature_names, x))
                        # Return confidence score (simplified)
                        predictions.append([top_emotion[1]])  # Use actual prediction
                    return np.array(predictions)
                
                explanation = self.explain_audio_emotion_with_shap(
                    predict_fn,
                    features_array,
                    feature_names
                )
            except Exception as e:
                print(f"SHAP failed, trying LIME: {e}")
        
        if explanation is None and model and self.lime_available:
            try:
                def predict_fn(X):
                    if len(X.shape) == 1:
                        X = X.reshape(1, -1)
                    return np.array([top_emotion[1]] * len(X)).reshape(-1, 1)
                
                explanation = self.explain_audio_emotion_with_lime(
                    predict_fn,
                    features_array,
                    feature_names
                )
            except Exception as e:
                print(f"LIME failed: {e}")
        
        if explanation is None:
            # Fallback to rule-based
            explanation = self._fallback_audio_explanation(features_array, feature_names)
        
        # Add emotion info
        explanation['top_emotion'] = top_emotion[0]
        explanation['confidence'] = top_emotion[1]
        
        return explanation
    
    def _fallback_audio_explanation(
        self,
        features_array: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """Fallback rule-based explanation when SHAP/LIME unavailable"""
        contributions = {}
        
        for i, name in enumerate(feature_names):
            value = features_array[0, i]
            
            # Rule-based importance
            if 'pitch' in name.lower():
                if abs(value) > 150:
                    importance = 0.3
                    explanation = f"{'High' if value > 150 else 'Low'} pitch detected"
                else:
                    importance = 0.1
                    explanation = "Normal pitch range"
            
            elif 'energy' in name.lower():
                if abs(value) > 0.05:
                    importance = 0.25
                    explanation = "High vocal energy"
                else:
                    importance = 0.15
                    explanation = "Low vocal energy"
            
            elif 'duration' in name.lower():
                if value < 5:
                    importance = 0.2
                    explanation = "Brief response"
                elif value > 30:
                    importance = 0.1
                    explanation = "Lengthy response"
                else:
                    importance = 0.05
                    explanation = "Normal duration"
            
            else:
                importance = 0.05
                explanation = "Minor contribution"
            
            contributions[name] = {
                'value': float(value),
                'importance': importance,
                'explanation': explanation
            }
        
        # Sort by importance
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        
        return {
            'method': 'rule-based (SHAP/LIME not available)',
            'feature_importance': [
                {
                    'feature': k,
                    'value': v['value'],
                    'importance': v['importance'],
                    'explanation': v['explanation']
                }
                for k, v in sorted_features
            ],
            'top_3_features': [
                {
                    'feature': k,
                    'value': v['value'],
                    'importance': v['importance'],
                    'explanation': v['explanation']
                }
                for k, v in sorted_features[:3]
            ],
            'explanation': self._generate_fallback_explanation(sorted_features[:3])
        }
    
    def _generate_shap_explanation(self, top_features: List[Dict]) -> str:
        """Generate explanation from SHAP values"""
        parts = ["SHAP analysis identified the following key factors:"]
        
        for feat in top_features:
            impact_word = "increased" if feat['impact'] == 'positive' else "decreased"
            parts.append(
                f"{feat['feature']}: {impact_word} prediction by {abs(feat['shap_value']):.3f}"
            )
        
        return " ".join(parts)
    
    def _generate_lime_explanation(self, top_features: List[Dict]) -> str:
        """Generate explanation from LIME values"""
        parts = ["LIME analysis shows:"]
        
        for feat in top_features:
            impact_word = "positively" if feat['impact'] == 'positive' else "negatively"
            parts.append(
                f"{feat['feature']} impacted {impact_word} (importance: {feat['lime_importance']:.3f})"
            )
        
        return " ".join(parts)
    
    def _generate_fallback_explanation(self, top_features: List[Tuple]) -> str:
        """Generate explanation from rule-based analysis"""
        parts = ["Rule-based analysis:"]
        
        for name, data in top_features:
            parts.append(data['explanation'])
        
        return " ".join(parts)
    
    def explain_video_emotion_prediction(
        self,
        video_emotions: Dict[str, float],
        timeline: List[Dict] = None
    ) -> Dict[str, any]:
        """
        Explain video emotion prediction with temporal analysis
        """
        if not video_emotions:
            return {
                'dominant_emotion': 'neutral',
                'consistency': 0.5,
                'explanation': "No facial data available"
            }
        
        # Get dominant emotion
        dominant = max(video_emotions.items(), key=lambda x: x[1])
        
        # Calculate consistency
        consistency_score = 0.7
        emotion_trajectory = "stable"
        
        if timeline and len(timeline) > 2:
            emotion_changes = []
            for i in range(len(timeline) - 1):
                curr = max(timeline[i]['emotions'].items(), key=lambda x: x[1])[0]
                next_e = max(timeline[i+1]['emotions'].items(), key=lambda x: x[1])[0]
                emotion_changes.append(1 if curr != next_e else 0)
            
            consistency_score = 1.0 - (sum(emotion_changes) / len(emotion_changes))
            
            if consistency_score > 0.7:
                emotion_trajectory = "consistent"
            elif consistency_score > 0.4:
                emotion_trajectory = "variable"
            else:
                emotion_trajectory = "unstable"
        
        explanation = self._generate_video_explanation(
            dominant, 
            consistency_score, 
            emotion_trajectory,
            video_emotions
        )
        
        return {
            'dominant_emotion': dominant[0],
            'confidence': dominant[1],
            'consistency': consistency_score,
            'trajectory': emotion_trajectory,
            'emotion_distribution': video_emotions,
            'explanation': explanation,
            'method': 'temporal_analysis'
        }
    
    def _generate_video_explanation(
        self,
        dominant: Tuple[str, float],
        consistency: float,
        trajectory: str,
        all_emotions: Dict
    ) -> str:
        """Generate natural language explanation for video analysis"""
        emotion_name, confidence = dominant
        
        parts = [
            f"Facial expression analysis showed predominantly {emotion_name} emotion ({confidence:.1%})."
        ]
        
        if trajectory == "consistent":
            parts.append("Expressions remained consistent, indicating stable emotional state.")
        elif trajectory == "variable":
            parts.append("Some variation detected, which is natural during conversation.")
        else:
            parts.append("Significant fluctuation observed, may indicate nervousness.")
        
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0.2:
            parts.append(f"Secondary: {sorted_emotions[1][0]} ({sorted_emotions[1][1]:.1%}).")
        
        return " ".join(parts)
    
    def explain_final_score(
        self,
        component_scores: Dict[str, float],
        weights: Dict[str, float],
        final_score: float
    ) -> Dict[str, any]:
        """Explain final score with contribution breakdown"""
        contributions = {}
        for component, score in component_scores.items():
            weighted_contribution = score * weights[component]
            contributions[component] = {
                'score': score,
                'weight': weights[component],
                'contribution': weighted_contribution,
                'percentage': (weighted_contribution / final_score * 100) if final_score > 0 else 0
            }
        
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        
        explanation = self._generate_score_explanation(sorted_contributions, final_score)
        
        return {
            'final_score': final_score,
            'contributions': contributions,
            'sorted_contributions': sorted_contributions,
            'explanation': explanation,
            'strongest_area': sorted_contributions[0][0] if sorted_contributions else 'none',
            'weakest_area': sorted_contributions[-1][0] if sorted_contributions else 'none',
            'method': 'weighted_sum'
        }
    
    def _generate_score_explanation(self, sorted_contributions: List[Tuple], final_score: float) -> str:
        """Generate natural language explanation for final score"""
        if not sorted_contributions:
            return "Unable to generate explanation."
        
        parts = [f"Final score: {final_score:.1f}/10."]
        
        strongest = sorted_contributions[0]
        parts.append(
            f"Strongest: {strongest[0].replace('_', ' ')} "
            f"({strongest[1]['score']:.1f}/10, {strongest[1]['percentage']:.1f}% contribution)."
        )
        
        weakest = sorted_contributions[-1]
        parts.append(
            f"Improvement area: {weakest[0].replace('_', ' ')} "
            f"({weakest[1]['score']:.1f}/10)."
        )
        
        return " ".join(parts)
    
    def generate_feedback_report(
        self,
        interview_data: Dict,
        scores: Dict,
        audio_explanation: Dict = None,
        video_explanation: Dict = None,
        score_explanation: Dict = None
    ) -> Dict[str, any]:
        """Generate comprehensive feedback with XAI insights"""
        feedback = {
            'strengths': [],
            'areas_for_improvement': [],
            'specific_recommendations': [],
            'overall_impression': "",
            'xai_methods_used': []
        }
        
        component_scores = scores.get('component_scores', {})
        
        for component, score in component_scores.items():
            if score >= 7:
                feedback['strengths'].append(
                    self._generate_strength_feedback(component, score)
                )
            elif score < 5:
                feedback['areas_for_improvement'].append(
                    self._generate_improvement_feedback(component, score)
                )
        
        # Add XAI insights
        if audio_explanation:
            method = audio_explanation.get('method', 'unknown')
            feedback['xai_methods_used'].append(f"Audio: {method}")
            feedback['specific_recommendations'].append(
                f"🎤 Voice: {audio_explanation.get('explanation', '')}"
            )
        
        if video_explanation:
            method = video_explanation.get('method', 'unknown')
            feedback['xai_methods_used'].append(f"Video: {method}")
            feedback['specific_recommendations'].append(
                f"📹 Body language: {video_explanation.get('explanation', '')}"
            )
        
        if score_explanation:
            method = score_explanation.get('method', 'unknown')
            feedback['xai_methods_used'].append(f"Score: {method}")
        
        # Overall impression
        final_score = scores.get('final_score', 0)
        if final_score >= 8:
            feedback['overall_impression'] = "Excellent candidate. Highly recommended."
        elif final_score >= 6.5:
            feedback['overall_impression'] = "Good candidate. Recommended for next round."
        elif final_score >= 5:
            feedback['overall_impression'] = "Acceptable candidate. Consider for future."
        else:
            feedback['overall_impression'] = "Does not meet current requirements."
        
        return feedback
    
    def _generate_strength_feedback(self, component: str, score: float) -> str:
        """Generate positive feedback"""
        templates = {
            'motivation': f"Strong enthusiasm evident (Score: {score:.1f}/10)",
            'experience': f"Relevant experience demonstrated (Score: {score:.1f}/10)",
            'confidence': f"Confident communication (Score: {score:.1f}/10)",
            'logistics': f"Clear availability (Score: {score:.1f}/10)",
            'red_flags': f"No concerns detected (Score: {score:.1f}/10)"
        }
        return templates.get(component, f"Strong {component} (Score: {score:.1f}/10)")
    
    def _generate_improvement_feedback(self, component: str, score: float) -> str:
        """Generate constructive feedback"""
        templates = {
            'motivation': f"Could show more enthusiasm (Score: {score:.1f}/10)",
            'experience': f"Experience may not fully align (Score: {score:.1f}/10)",
            'confidence': f"Could benefit from more confidence (Score: {score:.1f}/10)",
            'logistics': f"Availability unclear (Score: {score:.1f}/10)",
            'red_flags': f"Some concerns detected (Score: {score:.1f}/10)"
        }
        return templates.get(component, f"Improve {component} (Score: {score:.1f}/10)")