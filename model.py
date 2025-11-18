# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class EAGLE(nn.Module):
    """
    The PyTorch implementation of the EAGLE model.
    """
    def __init__(self, lmm_backbone_name, num_emotions, dropout=0.1):
        super().__init__()

        # 1. Load the LMM backbone (e.g., Qwen2-VL)
        print(f"Loading LMM backbone from {lmm_backbone_name}...")
        self.lmm_backbone = AutoModelForCausalLM.from_pretrained(
            lmm_backbone_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Get model dimensions
        d_model = self.lmm_backbone.config.hidden_size
        fused_dim = d_model + num_emotions

        # 2. Authenticity Head (Primary Task)
        self.authenticity_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 2)  # 0 for Human, 1 for LLM
        )

        # 3. Emotion Head (Auxiliary Task)
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_emotions)
        )

    def forward(self, input_ids, attention_mask, pixel_values, emotion_vectors_pe):
        """
        Forward pass for the EAGLE model.
        """
        # 1. Extract multimodal features (hmm) from the LMM backbone
        outputs = self.lmm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        last_hidden_states = outputs.hidden_states[-1]
        
        # Use the hidden state of the last non-padding token as the sentence representation (hmm)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        hmm = last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), sequence_lengths]

        # 2. Feature Fusion
        h_fused = torch.cat([hmm, emotion_vectors_pe], dim=-1)

        # 3. Pass through classification heads
        auth_logits = self.authenticity_head(h_fused)
        emotion_predictions = self.emotion_head(hmm)

        return auth_logits, emotion_predictions
