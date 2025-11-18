# affective_module.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AffectiveModule(nn.Module):
    """
    A wrapper for a pre-trained emotion classification model.
    This module is used to generate ground-truth emotion vectors for text.
    Its parameters are frozen during the training of the main EAGLE model.
    """
    def __init__(self, model_name, device):
        super(AffectiveModule, self).__init__()
        self.device = device
        
        print(f"Loading Affective Module from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Freeze the parameters of the affective model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts):
        """
        Takes a list of texts and returns their emotion vectors.
        The object is callable directly, e.g., affective_module(texts).
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        
        # Apply softmax to get probability distributions (emotion vectors)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return probabilities.cpu()

