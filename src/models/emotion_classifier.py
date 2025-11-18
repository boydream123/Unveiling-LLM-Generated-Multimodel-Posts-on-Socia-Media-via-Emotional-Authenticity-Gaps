"""
Emotion classification module for extracting affective representations.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union
import numpy as np


class EmotionClassifier(nn.Module):
    """
    Pre-trained emotion classifier for extracting emotional features.
    Based on RoBERTa fine-tuned on GoEmotions dataset.
    """
    
    def __init__(
        self, 
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        freeze: bool = True
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Emotion labels from GoEmotions dataset
        self.emotion_labels = [
            'joy', 'sadness', 'anger', 'fear', 
            'surprise', 'disgust', 'love'
        ]
        self.num_emotions = len(self.emotion_labels)
        
        # Classification head for emotions
        self.classifier = nn.Linear(self.model.config.hidden_size, self.num_emotions)
        
        if freeze:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze the backbone model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract emotion probabilities from text.
        
        Args:
            texts: Input text(s) to analyze
            
        Returns:
            Emotion probability scores of shape (batch_size, num_emotions)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(next(self.parameters()).device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Use [CLS] token representation
            cls_embeddings = hidden_states[:, 0, :]
        
        # Get emotion logits
        emotion_logits = self.classifier(cls_embeddings)
        
        # Apply sigmoid for multi-label emotion classification
        emotion_probs = torch.sigmoid(emotion_logits)
        
        return emotion_probs
    
    def get_emotion_vector(self, text: str) -> np.ndarray:
        """Get emotion vector for a single text."""
        self.eval()
        with torch.no_grad():
            emotion_probs = self.forward(text)
            return emotion_probs.cpu().numpy().squeeze()


class MultiEmotionClassifier(nn.Module):
    """
    Multiple emotion classifiers ensemble for robust emotion detection.
    """
    
    def __init__(self, model_names: List[str], freeze: bool = True):
        super().__init__()
        
        self.classifiers = nn.ModuleList([
            EmotionClassifier(name, freeze=freeze) for name in model_names
        ])
        
        # Determine total emotion dimensions
        self.total_emotion_dim = sum(
            classifier.num_emotions for classifier in self.classifiers
        )
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract emotion features from multiple classifiers.
        
        Args:
            texts: Input text(s) to analyze
            
        Returns:
            Concatenated emotion features of shape (batch_size, total_emotion_dim)
        """
        emotion_features = []
        
        for classifier in self.classifiers:
            emotion_probs = classifier(texts)
            emotion_features.append(emotion_probs)
        
        # Concatenate all emotion features
        return torch.cat(emotion_features, dim=-1)
