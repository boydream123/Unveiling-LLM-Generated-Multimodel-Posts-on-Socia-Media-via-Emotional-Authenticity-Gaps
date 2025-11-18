"""
EAGLE: Emotional Authenticity Gap-based LLM-Generated Content Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor, 
    Qwen2VLForConditionalGeneration,
    AutoTokenizer
)
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .emotion_classifier import EmotionClassifier, MultiEmotionClassifier


class EAGLE(nn.Module):
    """
    EAGLE framework for detecting LLM-generated multimodal content.
    
    This model implements the core methodology from the paper:
    "Unveiling LLM-Generated Multimodal Posts on Social Media via Emotional Authenticity Gaps"
    """
    
    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        hidden_dim: int = 4096,
        num_emotions: int = 7,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        lambda_emotion: float = 0.1
    ):
        super().__init__()
        
        self.lambda_emotion = lambda_emotion
        
        # Initialize multimodal backbone (MLLM)
        self.processor = AutoProcessor.from_pretrained(backbone_name)
        self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
            backbone_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # Initialize emotion classifier
        self.emotion_classifier = EmotionClassifier(
            model_name=emotion_model_name,
            freeze=True
        )
        
        # Feature dimensions
        self.multimodal_dim = hidden_dim
        self.emotion_dim = self.emotion_classifier.num_emotions
        self.fused_dim = self.multimodal_dim + self.emotion_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # Binary classification
        )
        
        # Emotion prediction head for auxiliary task
        self.emotion_projector = nn.Sequential(
            nn.Linear(self.multimodal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.emotion_dim)
        )
        
    def _freeze_backbone(self):
        """Freeze backbone model parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_multimodal_features(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> torch.Tensor:
        """
        Extract multimodal features using MLLM backbone.
        
        Args:
            images: Batch of images
            texts: Batch of texts
            
        Returns:
            Multimodal features of shape (batch_size, multimodal_dim)
        """
        # Prepare inputs for the processor
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.backbone.device)
        
        # Extract hidden states from backbone
        with torch.set_grad_enabled(not self.backbone.training):
            outputs = self.backbone.model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state of the first token ([CLS]-like)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            multimodal_features = hidden_states[:, 0, :]  # First token
        
        return multimodal_features
    
    def extract_emotion_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract emotion features from texts.
        
        Args:
            texts: Batch of texts
            
        Returns:
            Emotion features of shape (batch_size, emotion_dim)
        """
        return self.emotion_classifier(texts)
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: List[str],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of EAGLE model.
        
        Args:
            images: Batch of images of shape (batch_size, 3, H, W)
            texts: List of text strings
            labels: Ground truth labels for training
            
        Returns:
            Dictionary containing logits and losses
        """
        # Extract multimodal features
        h_mm = self.extract_multimodal_features(images, texts)
        
        # Extract emotion features
        p_e = self.extract_emotion_features(texts)
        
        # Feature fusion
        h_f = torch.cat([h_mm, p_e], dim=-1)
        
        # Classification
        logits = self.classifier(h_f)
        
        # Emotion prediction for auxiliary task
        emotion_pred = self.emotion_projector(h_mm)
        
        outputs = {
            'logits': logits,
            'emotion_pred': emotion_pred,
            'multimodal_features': h_mm,
            'emotion_features': p_e
        }
        
        # Calculate losses during training
        if labels is not None:
            # Primary authenticity classification loss
            auth_loss = F.cross_entropy(logits, labels)
            
            # Auxiliary emotion reconstruction loss
            emotion_loss = F.mse_loss(emotion_pred, p_e.detach())
            
            # Combined loss
            total_loss = auth_loss + self.lambda_emotion * emotion_loss
            
            outputs.update({
                'loss': total_loss,
                'auth_loss': auth_loss,
                'emotion_loss': emotion_loss
            })
        
        return outputs
    
    def predict(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on new data.
        
        Args:
            images: Batch of images
            texts: List of text strings
            
        Returns:
            Predicted labels and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, texts)
            logits = outputs['logits']
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions, probs


class EAGLEWithDifferentBackbones(EAGLE):
    """
    EAGLE framework that supports different MLLM backbones.
    """
    
    SUPPORTED_BACKBONES = {
        'qwen2-vl': "Qwen/Qwen2.5-VL-7B-Instruct",
        'qwen2-vl-2b': "Qwen/Qwen2.5-VL-2B-Instruct", 
        'llava': "llava-hf/llava-1.5-7b-hf",
        'llava-13b': "llava-hf/llava-1.5-13b-hf"
    }
    
    def __init__(
        self, 
        backbone_type: str = 'qwen2-vl',
        **kwargs
    ):
        if backbone_type not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. "
                f"Supported types: {list(self.SUPPORTED_BACKBONES.keys())}"
            )
        
        backbone_name = self.SUPPORTED_BACKBONES[backbone_type]
        super().__init__(backbone_name=backbone_name, **kwargs)
        
        self.backbone_type = backbone_type


def create_eagle_model(config) -> EAGLE:
    """
    Factory function to create EAGLE model from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized EAGLE model
    """
    return EAGLE(
        backbone_name=config.model.backbone_name,
        emotion_model_name=config.model.emotion_model_name,
        hidden_dim=config.model.hidden_dim,
        num_emotions=config.model.num_emotions,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
        lambda_emotion=config.training.lambda_emotion
    )
