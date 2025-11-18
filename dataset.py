# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class EAGLEDataset(Dataset):
    """
    Custom dataset for the EAGLE model.
    It loads images, text, and pre-computes emotion vectors.
    """
    def __init__(self, data_samples, lmm_processor, affective_module):
        """
        Args:
            data_samples (list): A list of dicts, each like:
                                 {'image_path': str, 'text': str, 'label': int}
            lmm_processor: The processor for the LMM backbone.
            affective_module: An instance of the AffectiveModule.
        """
        self.data_samples = data_samples
        self.processor = lmm_processor
        
        # Pre-compute all emotion vectors (pgt) to speed up training
        print("Pre-computing emotion vectors for the dataset...")
        all_texts = [sample['text'] for sample in self.data_samples]
        self.emotion_vectors = self._compute_emotion_vectors(all_texts, affective_module)

    def _compute_emotion_vectors(self, texts, affective_module, batch_size=32):
        emotion_vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing Emotions"):
            batch_texts = texts[i:i + batch_size]
            # Using logits provides a richer, uncompressed signal for the MSE loss
            logits = affective_module(batch_texts).cpu()
            emotion_vectors.append(logits)
        return torch.cat(emotion_vectors, dim=0)

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        image_path = sample['image_path']
        text = sample['text']
        label = sample['label']
        
        image = Image.open(image_path).convert("RGB")
        
        # Format input according to the LMM's chat template
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor([prompt], images=[image], return_tensors="pt", padding=True)

        emotion_ground_truth = self.emotion_vectors[idx]

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0).to(torch.bfloat16),
            "authenticity_label": torch.tensor(label, dtype=torch.long),
            "emotion_ground_truth": emotion_ground_truth.float()
        }
