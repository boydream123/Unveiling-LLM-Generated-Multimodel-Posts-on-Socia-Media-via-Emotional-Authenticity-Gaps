# model.py (Corrected for newer models like Qwen2.5-VL)
import torch
import torch.nn as nn
# CHANGE 1: Import the more general AutoModel instead of the specific AutoModelForCausalLM
from transformers import AutoModel 

class EAGLE(nn.Module):
    """
    The PyTorch implementation of the EAGLE model.
    Updated to be compatible with newer VLMs like Qwen2.5-VL.
    """
    def __init__(self, lmm_backbone_name, num_emotions, dropout=0.1):
        super().__init__()

        print(f"Loading LMM backbone from {lmm_backbone_name}...")
        # CHANGE 2: Use AutoModel, which is more flexible for different architectures
        # CHANGE 3: Use the modern `dtype` argument instead of the deprecated `torch_dtype`
        self.lmm_backbone = AutoModel.from_pretrained(
            lmm_backbone_name,
            dtype=torch.bfloat16, # Use 'dtype' instead of 'torch_dtype'
            device_map="auto",
            trust_remote_code=True,
        )

        # Get model dimensions
        d_model = self.lmm_backbone.config.hidden_size
        fused_dim = d_model + num_emotions

        # Authenticity Head (Primary Task)
        self.authenticity_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 2)
        )

        # Emotion Head (Auxiliary Task)
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_emotions)
        )

    def forward(self, input_ids, attention_mask, pixel_values, emotion_vectors_pe):
        # The Qwen2.5-VL model requires pixel_values as a list
        # We need to handle the batched tensor correctly
        b, c, h, w = pixel_values.shape
        pixel_values_list = [pv for pv in pixel_values]

        outputs = self.lmm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_list, # Pass pixel_values as a list
            output_hidden_states=True
        )
        
        last_hidden_states = outputs.hidden_states[-1]
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        hmm = last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), sequence_lengths]

        h_fused = torch.cat([hmm, emotion_vectors_pe], dim=-1)

        auth_logits = self.authenticity_head(h_fused)
        emotion_predictions = self.emotion_head(hmm)

        return auth_logits, emotion_predictions
