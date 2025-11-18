# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW  # CORRECTED: Import AdamW from torch.optim
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import argparse
import os
import random
import pandas as pd

from model import EAGLE
from dataset import EAGLEDataset
from affective_module import AffectiveModule

def load_mvsa_samples(data_dir, label_file, text_dir, llm_fraction=0.5):
    """
    Loads and prepares the MVSA dataset samples, simulating a mixed human/LLM scenario.
    """
    print("Loading MVSA dataset...")
    samples = []
    
    try:
        labels_df = pd.read_csv(label_file, header=None, names=['id', 'sentiment'])
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file}")
        print("Please download the MVSA dataset and place it correctly.")
        return []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing MVSA data"):
        image_id = row['id']
        image_path = os.path.join(data_dir, image_id)
        text_path = os.path.join(text_dir, image_id.replace('.jpg', '.txt'))
        
        if os.path.exists(image_path) and os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                if text:
                    samples.append({'image_path': image_path, 'text': text})
            except Exception as e:
                print(f"Could not read {text_path}: {e}")
    
    if not samples:
        print("No valid samples were loaded. Please check the dataset paths.")
        return []
        
    print(f"Loaded {len(samples)} valid image-text pairs from MVSA.")

    # Simulate the authenticity labels by randomly assigning half as 'LLM-generated'
    random.shuffle(samples)
    num_llm = int(len(samples) * llm_fraction)
    for i in range(len(samples)):
        samples[i]['label'] = 1 if i < num_llm else 0
            
    print(f"Simulated authenticity labels: {len(samples) - num_llm} Human, {num_llm} LLM.")
    return samples

def train(args):
    # --- 1. Setup and Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(args.lmm_model_name, trust_remote_code=True)
    affective_module = AffectiveModule(device="cpu")
    num_emotions = affective_module.model.config.num_labels
    model = EAGLE(lmm_backbone_name=args.lmm_model_name, num_emotions=num_emotions)
    model.authenticity_head.to(device)
    model.emotion_head.to(device)

    # --- 2. Data Preparation using MVSA ---
    all_samples = load_mvsa_samples(
        data_dir=os.path.join(args.dataset_path, "data"),
        label_file=os.path.join(args.dataset_path, "labelResult_single.txt"),
        text_dir=os.path.join(args.dataset_path, "data")
    )
    if not all_samples: return

    train_size = int(0.9 * len(all_samples))
    val_size = len(all_samples) - train_size
    train_samples, val_samples = random_split(all_samples, [train_size, val_size])

    train_dataset = EAGLEDataset(list(train_samples), processor, affective_module)
    val_dataset = EAGLEDataset(list(val_samples), processor, affective_module)

    collate_fn = lambda batch: {key: torch.stack([d[key] for d in batch]) for key in batch[0]}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 3. Loss Functions and Optimizer ---
    loss_fn_auth = nn.CrossEntropyLoss()
    loss_fn_emo = nn.MSELoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # --- 4. Training & Validation Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            # Move data to device and perform a training step
            input_ids, attention_mask, pixel_values, auth_labels, emotion_gts = (
                batch['input_ids'].to(device), batch['attention_mask'].to(device),
                batch['pixel_values'].to(device), batch['authenticity_label'].to(device),
                batch['emotion_ground_truth'].to(device)
            )
            auth_logits, emotion_preds = model(input_ids, attention_mask, pixel_values, emotion_gts.to(model.lmm_backbone.dtype))
            loss_auth = loss_fn_auth(auth_logits, auth_labels)
            loss_emo = loss_fn_emo(emotion_preds, emotion_gts)
            total_loss = loss_auth + args.lambda_emo * loss_emo
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training finished. Average Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validate]"):
                input_ids, attention_mask, pixel_values, auth_labels, emotion_gts = (
                    batch['input_ids'].to(device), batch['attention_mask'].to(device),
                    batch['pixel_values'].to(device), batch['authenticity_label'].to(device),
                    batch['emotion_ground_truth'].to(device)
                )
                auth_logits, emotion_preds = model(input_ids, attention_mask, pixel_values, emotion_gts.to(model.lmm_backbone.dtype))
                loss_auth = loss_fn_auth(auth_logits, auth_labels)
                loss_emo = loss_fn_emo(emotion_preds, emotion_gts)
                total_val_loss += (loss_auth + args.lambda_emo * loss_emo).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation finished. Average Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save({'authenticity_head': model.authenticity_head.state_dict(), 'emotion_head': model.emotion_head.state_dict()}, args.save_path)
                print(f"New best model saved to {args.save_path} (Val Loss: {best_val_loss:.4f})")

    print("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAGLE model with a flexible LMM backbone on the MVSA dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root of the MVSA-Corpus directory")
    parser.add_argument("--lmm_model_name", type=str, default="Qwen/Qwen2-VL-1.5B-Instruct", help="LMM backbone model name from Hugging Face Hub")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (keep small for large models)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--lambda_emo", type=float, default=0.1, help="Weight for the auxiliary emotion loss term")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_eagle_heads.pth", help="Path to save the best trained model heads")
    
    args = parser.parse_args()
    train(args)
