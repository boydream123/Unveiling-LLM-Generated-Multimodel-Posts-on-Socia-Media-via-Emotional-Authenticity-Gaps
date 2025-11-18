#!/usr/bin/env python3
"""
Training script for EAGLE multimodal detection framework.
"""
import argparse
import torch
import random
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import EAGLEConfig
from src.models.eagle import create_eagle_model
from src.data.dataset import EAGLEDataset, create_dataloaders
from src.training.trainer import EAGLETrainer
from torchvision import transforms


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_transforms(config):
    """Create image transformations."""
    return transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def main():
    parser = argparse.ArgumentParser(description='Train EAGLE model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data CSV/JSON')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data CSV/JSON')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data CSV/JSON')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = EAGLEConfig.from_yaml(args.config)
    config.data.train_data_path = args.train_data
    config.data.val_data_path = args.val_data
    config.data.test_data_path = args.test_data
    config.output_dir = args.output_dir
    
    # Set seed
    set_seed(config.seed)
    
    # Create transforms
    transform = create_transforms(config)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = EAGLEDataset(
        args.train_data, args.image_dir, transform=transform
    )
    
    val_dataset = EAGLEDataset(
        args.val_data, args.image_dir, transform=transform
    )
    
    test_dataset = None
    if args.test_data:
        test_dataset = EAGLEDataset(
            args.test_data, args.image_dir, transform=transform
        )
    
    # Create data loaders
    dataloaders = create_dataloaders(
        config, train_dataset, val_dataset, test_dataset
    )
    
    # Create model
    print("Creating EAGLE model...")
    model = create_eagle_model(config)
    
    # Create trainer
    trainer = EAGLETrainer(
        model=model,
        config=config,
        train_loader=dataloaders.get('train'),
        val_loader=dataloaders.get('val'),
        test_loader=dataloaders.get('test')
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
    print("Starting training...")
    final_metrics = trainer.train()
    
    print("Training completed!")
    print("Final metrics:", final_metrics)


if __name__ == '__main__':
    main()
