#!/usr/bin/env python3
"""
Evaluation script for EAGLE multimodal detection framework.
"""
import argparse
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import EAGLEConfig
from src.models.eagle import create_eagle_model
from src.data.dataset import EAGLEDataset, EAGLECollator
from src.utils.metrics import EAGLEMetrics, evaluate_cross_domain
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description='Evaluate EAGLE model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV/JSON')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--cross_domain', type=str, default=None,
                       help='Path to cross-domain test data for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = EAGLEConfig.from_yaml(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = EAGLEDataset(
        args.test_data, args.image_dir, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=EAGLECollator()
    )
    
    # Load model
    print("Loading EAGLE model...")
    model = create_eagle_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Evaluate
    print("Evaluating model...")
    metrics = EAGLEMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            
            outputs = model(images, texts)
            predictions = torch.argmax(outputs['logits'], dim=-1)
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            
            metrics.update(predictions, labels, probabilities)
    
    # Compute metrics
    results = metrics.compute()
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix
    fig = metrics.plot_confusion_matrix()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Save classification report
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(metrics.get_classification_report())
    
    # Cross-domain evaluation if specified
    if args.cross_domain:
        print("Performing cross-domain evaluation...")
        
        cross_domain_dataset = EAGLEDataset(
            args.cross_domain, args.image_dir, transform=transform
        )
        
        cross_domain_loader = DataLoader(
            cross_domain_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=EAGLECollator()
        )
        
        cross_domain_results = evaluate_cross_domain(
            model, test_loader, cross_domain_loader, device
        )
        
        print("Cross-domain Results:")
        print(json.dumps(cross_domain_results, indent=2))
        
        with open(output_dir / 'cross_domain_results.json', 'w') as f:
            json.dump(cross_domain_results, f, indent=2)
    
    print(f"Evaluation completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
