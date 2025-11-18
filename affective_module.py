# affective_module.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AffectiveModule(nn.Module):
    """
    A wrapper for a pre-trained emotion classifier.
    The weights of this module will be frozen during the EAGLE training process.
    """
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", device="cpu"):
        """
        Args:
            model_name (str): The name of the emotion classification model on the Hugging Face Hub.
            device (str): The device to load the model on.
        """
        super().__init__()
        print(f"Loading Affective Module from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode permanently

    @torch.no_grad()
    def forward(self, texts):
        """
        Extracts emotion logits from a batch of texts.

        Args:
            texts (list[str]): A list of text strings.

        Returns:
            torch.Tensor: Emotion logits with shape [batch_size, num_emotions].
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.model.device)

        outputs = self.model(**inputs)
        return outputs.logits
