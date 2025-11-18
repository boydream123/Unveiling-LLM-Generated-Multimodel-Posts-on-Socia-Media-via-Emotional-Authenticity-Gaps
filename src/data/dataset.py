"""
Dataset classes for EAGLE multimodal detection.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json


class EAGLEDataset(Dataset):
    """
    Dataset for EAGLE multimodal detection training and evaluation.
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        transform=None,
        max_text_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to CSV/JSON file containing data
            image_dir: Directory containing images
            transform: Image transformation pipeline
            max_text_length: Maximum text sequence length
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.max_text_length = max_text_length
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data_list = json.load(f)
            self.data = pd.DataFrame(data_list)
        else:
            raise ValueError("Data file must be CSV or JSON format")
        
        # Validate required columns
        required_columns = ['image_path', 'text', 'label']
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: PIL Image or transformed tensor
                - text: Text string
                - label: Binary label (0: human, 1: LLM-generated)
        """
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Handle corrupted images with a default image
            image = Image.new('RGB', (224, 224), color='white')
            print(f"Warning: Could not load image {image_path}, using default. Error: {e}")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Get text and label
        text = str(row['text'])
        label = int(row['label'])
        
        return {
            'image': image,
            'text': text,
            'label': label,
            'image_path': str(image_path)
        }


class EAGLECollator:
    """
    Custom collator for batching EAGLE dataset samples.
    """
    
    def __init__(self, processor=None):
        self.processor = processor
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from EAGLEDataset
            
        Returns:
            Batched data ready for model input
        """
        # Extract components
        images = [sample['image'] for sample in batch]
        texts = [sample['text'] for sample in batch]
        labels = [sample['label'] for sample in batch]
        
        # Stack images if they're tensors, otherwise keep as list for processor
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'images': images,
            'texts': texts,
            'labels': labels
        }


def create_dataloaders(
    config,
    train_dataset: Optional[EAGLEDataset] = None,
    val_dataset: Optional[EAGLEDataset] = None,
    test_dataset: Optional[EAGLEDataset] = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Data configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        
    Returns:
        Dictionary of data loaders
    """
    collator = EAGLECollator()
    dataloaders = {}
    
    if train_dataset:
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    if val_dataset:
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    if test_dataset:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    return dataloaders


# Example data preparation functions
def prepare_twibot22_dataset(data_dir: str) -> Tuple[EAGLEDataset, EAGLEDataset, EAGLEDataset]:
    """Prepare TwiBot-22 dataset splits."""
    # Implementation would depend on TwiBot-22 data format
    pass


def prepare_social_media_dataset(data_dir: str) -> Tuple[EAGLEDataset, EAGLEDataset, EAGLEDataset]:
    """Prepare Social Media dataset splits."""
    # Implementation would depend on Social Media dataset format
    pass


def prepare_newsclippings_dataset(data_dir: str) -> Tuple[EAGLEDataset, EAGLEDataset, EAGLEDataset]:
    """Prepare NewsCLIPpings dataset splits."""
    # Implementation would depend on NewsCLIPpings data format
    pass
