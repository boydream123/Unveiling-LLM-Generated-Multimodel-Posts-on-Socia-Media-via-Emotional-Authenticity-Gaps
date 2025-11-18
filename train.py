# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AdamW
from PIL import Image
from tqdm import tqdm
import argparse
import os

from model import EAGLE
from dataset import EAGLEDataset
from affective_module import AffectiveModule

def train(args):
    # --- 1. Setup and Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(args.lmm_model_name, trust_remote_code=True)
    
    # Affective module is used for dataset pre-computation, can run on CPU
    affective_module = AffectiveModule(device="cpu")
    num_emotions = affective_module.model.config.num_labels
    
    model = EAGLE(lmm_backbone_name=args.lmm_model_name, num_emotions=num_emotions)
    model.authenticity_head.to(device)
    model.emotion_head.to(device)

    # --- 2. Data Preparation ---
    # Create dummy data for demonstration. Replace with your actual data loading logic.
    print("Creating dummy data for demonstration...")
    os.makedirs("dummy_data", exist_ok=True)
    Image.new('RGB', (224, 224), color='red').save("dummy_data/human.jpg")
    Image.new('RGB', (224, 224), color='blue').save("dummy_data/llm.jpg")
    
    train_samples = [
        {'image_path': 'dummy_data/human.jpg', 'text': 'Graduation day! So thrilled for the future, yet so sad to say goodbye.', 'label': 0},
        {'image_path': 'dummy_data/llm.jpg', 'text': 'A joyous graduation ceremony event took place.', 'label': 1},
    ] * 10  # Duplicate for a larger dummy dataset

    train_dataset = EAGLEDataset(train_samples, processor, affective_module)
    collate_fn = lambda batch: {key: torch.stack([d[key] for d in batch]) for key in batch[0]}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # --- 3. Loss Functions and Optimizer ---
    loss_fn_auth = nn.CrossEntropyLoss()
    loss_fn_emo = nn.MSELoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # --- 4. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            auth_labels = batch['authenticity_label'].to(device)
            emotion_gts = batch['emotion_ground_truth'].to(device)

            # Forward pass
            auth_logits, emotion_preds = model(
                input_ids, attention_mask, pixel_values,
                emotion_gts.to(model.lmm_backbone.dtype)
            )
            
            # Calculate loss
            loss_auth = loss_fn_auth(auth_logits, auth_labels)
            loss_emo = loss_fn_emo(emotion_preds, emotion_gts)
            total_loss = loss_auth + args.lambda_emo * loss_emo
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            
        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("--- Training Finished ---")
    
    # Save the trainable parts of the model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'authenticity_head': model.authenticity_head.state_dict(),
            'emotion_head': model.emotion_head.state_dict(),
        }, args.save_path)
        print(f"Trainable model heads saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAGLE model with a flexible LMM backbone")
    parser.add_argument("--lmm_model_name", type=str, default="Qwen/Qwen2-VL-1.5B-Instruct", help="LMM backbone model name from Hugging Face Hub")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (keep small for large models)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--lambda_emo", type=float, default=0.1, help="Weight for the auxiliary emotion loss term")
    parser.add_argument("--save_path", type=str, default="checkpoints/eagle_heads.pth", help="Path to save the trained model heads")
    
    args = parser.parse_args()
    train(args)
