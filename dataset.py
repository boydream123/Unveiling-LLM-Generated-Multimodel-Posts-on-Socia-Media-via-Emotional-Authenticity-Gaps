# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class EAGLEDataset(Dataset):
    """
    Custom PyTorch Dataset for the EAGLE model.
    It loads data samples and pre-computes ground-truth emotion vectors.
    Crucially, it returns raw data (prompts, images), leaving the final
    tensor conversion and padding to a custom `collate_fn` in the training script.
    This enables efficient dynamic padding for variable-length text sequences.
    """
    def __init__(self, samples, processor, affective_module):
        self.samples = samples
        self.processor = processor
        self.affective_module = affective_module
        
        print("Pre-computing emotion vectors for the dataset...")
        self.emotion_vectors = []
        texts_to_process = [sample['text'] for sample in self.samples]
        
        # Process in batches for efficiency
        batch_size = 32 
        for i in tqdm(range(0, len(texts_to_process), batch_size), desc="Computing Emotions"):
            batch_texts = texts_to_process[i:i + batch_size]
            # Call the affective_module object directly
            batch_vectors = self.affective_module(batch_texts)
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
            # If an image is broken, get the next one
            return self.__getitem__((idx + 1) % len(self))

        # Construct the prompt in the format expected by the LMM
        prompt = f"<|im_start|>user\nPicture 1:<img>{image_path}</img>\n{text}<|im_end|><|im_start|>assistant\n"
        
        # Return raw data; the collate_fn will handle batch processing
        return {
            'prompt': prompt,
            'image': image,
            'authenticity_label': torch.tensor(authenticity_label, dtype=torch.long),
            'emotion_ground_truth': emotion_gt.clone().detach()
        }

