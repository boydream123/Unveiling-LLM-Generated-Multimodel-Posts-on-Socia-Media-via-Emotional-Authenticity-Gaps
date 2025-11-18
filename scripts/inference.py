#!/usr/bin/env python3
"""
Inference script for EAGLE multimodal detection framework.
"""
import argparse
import torch
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import EAGLEConfig
from src.models.eagle import create_eagle_model
from torchvision import transforms


def load_model(config_path: str, checkpoint_path: str):
    """Load trained EAGLE model."""
    config = EAGLEConfig.from_yaml(config_path)
    model = create_eagle_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, device, config


def preprocess_image(image_path: str, image_size: int = 224):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict_single(
    model, 
    device, 
    image_path: str, 
    text: str, 
    config
) -> dict:
    """Make prediction on a single image-text pair."""
    # Preprocess image
    image = preprocess_image(image_path, config.data.image_size).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image, [text])
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
    
    # Convert to human-readable format
    label_map = {0: 'Human-authored', 1: 'LLM-generated'}
    
    result = {
        'prediction': label_map[prediction.item()],
        'confidence': probabilities.max().item(),
        'probabilities': {
            'human': probabilities[0, 0].item(),
            'llm_generated': probabilities[0, 1].item()
        },
        'image_path': image_path,
        'text': text
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='EAGLE inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--text', type=str, required=True,
                       help='Input text')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction results (JSON)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading EAGLE model...")
    model, device, config = load_model(args.config, args.checkpoint)
    
    # Make prediction
    print("Making prediction...")
    result = predict_single(model, device, args.image, args.text, config)
    
    # Print results
    print("\n" + "="*50)
    print("EAGLE Detection Results")
    print("="*50)
    print(f"Image: {result['image_path']}")
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Human probability: {result['probabilities']['human']:.4f}")
    print(f"LLM probability: {result['probabilities']['llm_generated']:.4f}")
    print("="*50)
    
    # Save results if output path specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
