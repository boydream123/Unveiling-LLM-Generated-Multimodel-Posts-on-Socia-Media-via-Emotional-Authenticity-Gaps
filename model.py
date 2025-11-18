# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class EAGLEModel(nn.Module):
    """
    The main EAGLE model which integrates a Large Multimodal Model (LMM)
    with two custom prediction heads: one for authenticity classification
    and one for emotion prediction.
    """
    def __init__(self, lmm_model_name, num_emotions=28, freeze_lmm=True):
        super(EAGLEModel, self).__init__()
        
        print(f"Loading LMM backbone from {lmm_model_name}...")
        self.lmm = AutoModelForCausalLM.from_pretrained(
            lmm_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )

        if freeze_lmm:
            print("Freezing LMM parameters...")
            for param in self.lmm.parameters():
                param.requires_grad = False
        
        # The hidden size of the LMM (e.g., 4096 for LLaMA-7B)
        hidden_size = self.lmm.config.hidden_size
        
        # Define the prediction heads
        self.authenticity_head = nn.Linear(hidden_size, 2)  # Binary classification: Human vs. LLM
        self.emotion_head = nn.Linear(hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get the outputs from the LMM backbone
        # We need the hidden states, not the language modeling logits
        outputs = self.lmm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # Get the hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Use the hidden state of the last token as the pooled representation
        # This is a common strategy for sequence classification with causal LMs
        # Shape: (batch_size, sequence_length, hidden_size)
        # We take the features corresponding to the last non-padding token
        sequence_lengths = torch.ne(input_ids, self.lmm.config.pad_token_id).sum(-1) - 1
        pooled_features = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), sequence_lengths]

        # Pass the pooled features through the prediction heads
        auth_logits = self.authenticity_head(pooled_features)
        emotion_logits = self.emotion_head(pooled_features)
        
        return {
            'authenticity_logits': auth_logits,
            'emotion_logits': emotion_logits
        }

