# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
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

def load_flickr30k_samples(data_dir, llm_fraction=0.5):
    """
    Loads and prepares the Flickr30k dataset samples.
    Each image has 5 captions, we treat each image-caption pair as a unique sample.
    """
    print("Loading Flickr30k dataset...")
    samples = []
    
    images_dir = os.path.join(data_dir, "images")
    captions_file = os.path.join(data_dir, "captions.token")

    if not os.path.isdir(images_dir) or not os.path.isfile(captions_file):
        print(f"Error: Flickr30k data not found in {data_dir}.")
        print("Please download the data and organize it as specified in the instructions.")
        return []

    # Read captions file
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = f.readlines()

    for line in tqdm(captions_data, desc="Processing Flickr30k data"):
        parts = line.strip().split('\t')
        image_name_part, caption = parts[0], parts[1]
        image_name = image_name_part.split('#')[0]
        
        image_path = os.path.join(images_dir, image_name)
        
        if os.path.exists(image_path):
            samples.append({'image_path': image_path, 'text': caption})

    if not samples:
        print("No valid samples were loaded. Please check the dataset paths.")
        return []
        
    print(f"Loaded {len(samples)} valid image-caption pairs from Flickr30k.")

    # Simulate authenticity labels (same logic as before)
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

    # --- 2. Data Preparation using Flickr30k ---
    all_samples = load_flickr30k_samples(
        data_dir=args.dataset_path,
    )
    if not all_samples: return

    # For faster experimentation, you can use a subset of the data
    if args.subset_size > 0:
        print(f"Using a subset of {args.subset_size} samples for training.")
        random.shuffle(all_samples)
        all_samples = all_samples[:args.subset_size]

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
    parser = argparse.ArgumentParser(description="Train EAGLE model with a flexible LMM backbone on the Flickr30k dataset")
    parser.add_argument("--dataset_path", type=str, default="./flickr30k_data", help="Path to the directory containing Flickr30k data (images/ and captions.token)")
    parser.add_argument("--lmm_model_name", type=str, default="Qwen/Qwen2-VL-1.5B-Instruct", help="LMM backbone model name from Hugging Face Hub")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add-argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lambda_emo", type=float, default=0.1, help="Weight for the auxiliary emotion loss")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_eagle_flickr30k.pth", help="Path to save the best model heads")
    parser.add_argument("--subset_size", type=int, default=1000, help="Use a smaller subset of the dataset for quick testing. Set to -1 to use all data.")
    
    args = parser.parse_args()
    train(args)
