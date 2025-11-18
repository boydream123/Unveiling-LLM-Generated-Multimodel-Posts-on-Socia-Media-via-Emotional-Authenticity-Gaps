# dataset.py (Final Corrected Version)
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class EAGLEDataset(Dataset):
    """
    Custom PyTorch Dataset for the EAGLE model.
    Handles image loading, text processing, and pre-computation of emotion vectors.
    """
    def __init__(self, samples, processor, affective_module):
        self.samples = samples
        self.processor = processor
        self.affective_module = affective_module
        
        print("Pre-computing emotion vectors for the dataset...")
        self.emotion_vectors = []
        texts_to_process = [sample['text'] for sample in self.samples]
        
        batch_size = 32 
        for i in tqdm(range(0, len(texts_to_process), batch_size), desc="Computing Emotions"):
            batch_texts = texts_to_process[i:i + batch_size]
            with torch.no_grad():
                # --- THIS IS THE CORRECTED LINE ---
                # Call the affective_module object directly, as it has a __call__ method.
                batch_vectors = self.affective_module(batch_texts)
                # --- END OF CORRECTION ---
            self.emotion_vectors.extend(batch_vectors)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        text = sample['text']
        authenticity_label = sample['label']
        emotion_gt = self.emotion_vectors[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        prompt = f"<|im_start|>user\nPicture 1:<img>{image_path}</img>\n{text}<|im_end|><|im_start|>assistant\n"
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'authenticity_label': torch.tensor(authenticity_label, dtype=torch.long),
            'emotion_ground_truth': emotion_gt.clone().detach()
        }
