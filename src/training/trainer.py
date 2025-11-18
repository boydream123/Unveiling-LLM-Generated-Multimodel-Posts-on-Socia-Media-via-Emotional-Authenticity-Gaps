"""
Training pipeline for EAGLE framework.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import wandb

from ..models.eagle import EAGLE
from ..utils.metrics import EAGLEMetrics
from ..utils.config import EAGLEConfig


class EAGLETrainer:
    """
    Trainer class for EAGLE multimodal detection model.
    """
    
    def __init__(
        self,
        model: EAGLE,
        config: EAGLEConfig,
        train_loader=None,
        val_loader=None,
        test_loader=None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = 0.0
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project="eagle-multimodal-detection",
                config=config.__dict__,
                name=f"eagle-{config.model.backbone_name.replace('/', '-')}"
            )
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters for different learning rates if needed
        optimizer_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'backbone' not in n and p.requires_grad],
                'lr': self.config.training.learning_rate
            }
        ]
        
        # Lower learning rate for backbone if not frozen
        backbone_params = [p for n, p in self.model.named_parameters() 
                          if 'backbone' in n and p.requires_grad]
        if backbone_params:
            optimizer_params.append({
                'params': backbone_params,
                'lr': self.config.training.learning_rate * 0.1  # 10x lower LR for backbone
            })
        
        return AdamW(
            optimizer_params,
            weight_decay=self.config.training.weight_decay
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.train_loader is None:
            return None
        
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_auth_loss = 0.0
        total_emotion_loss = 0.0
        metrics = EAGLEMetrics()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['images'].to(self.device)
            texts = batch['texts']
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, texts, labels)
            loss = outputs['loss']
            auth_loss = outputs['auth_loss']
            emotion_loss = outputs['emotion_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            predictions = torch.argmax(outputs['logits'], dim=-1)
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            metrics.update(predictions, labels, probabilities)
            
            # Accumulate losses
            total_loss += loss.item()
            total_auth_loss += auth_loss.item()
            total_emotion_loss += emotion_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'auth_loss': f'{auth_loss.item():.4f}',
                'emo_loss': f'{emotion_loss.item():.4f}'
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_training_step(loss.item(), auth_loss.item(), emotion_loss.item())
        
        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        epoch_metrics.update({
            'loss': total_loss / len(self.train_loader),
            'auth_loss': total_auth_loss / len(self.train_loader),
            'emotion_loss': total_emotion_loss / len(self.train_loader)
        })
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        metrics = EAGLEMetrics()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images, texts, labels)
                loss = outputs['loss']
                
                # Update metrics
                predictions = torch.argmax(outputs['logits'], dim=-1)
                probabilities = torch.softmax(outputs['logits'], dim=-1)
                metrics.update(predictions, labels, probabilities)
                
                total_loss += loss.item()
        
        # Compute validation metrics
        val_metrics = metrics.compute()
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        return val_metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model."""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        metrics = EAGLEMetrics()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                images = batch['images'].to(self.device)
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images, texts, labels)
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                probabilities = torch.softmax(outputs['logits'], dim=-1)
                metrics.update(predictions, labels, probabilities)
        
        test_metrics = metrics.compute()
        
        # Save detailed test results
        self._save_test_results(metrics)
        
        return test_metrics
    
    def train(self) -> Dict[str, float]:
        """Full training pipeline."""
        self.logger.info("Starting EAGLE training...")
        
        best_metrics = {}
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch} Training - {train_metrics}")
            
            # Validation
            if self.val_loader and epoch % self.config.eval_steps == 0:
                val_metrics = self.validate()
                self.logger.info(f"Epoch {epoch} Validation - {val_metrics}")
                
                # Save best model
                current_metric = val_metrics.get('f1_macro', 0.0)
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.save_checkpoint('best_model.pt')
                    best_metrics = val_metrics.copy()
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()},
                        'epoch': epoch
                    })
            
            # Save periodic checkpoint
            if epoch % self.config.save_steps == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Final testing
        if self.test_loader:
            test_metrics = self.test()
            self.logger.info(f"Final Test Results - {test_metrics}")
            best_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        self.logger.info("Training completed!")
        return best_metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.output_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        self.logger.info(f"Checkpoint loaded: {filename}")
    
    def _log_training_step(self, loss: float, auth_loss: float, emotion_loss: float):
        """Log training step metrics."""
        if self.config.use_wandb:
            wandb.log({
                'step_loss': loss,
                'step_auth_loss': auth_loss,
                'step_emotion_loss': emotion_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'global_step': self.global_step
            })
    
    def _save_test_results(self, metrics: EAGLEMetrics):
        """Save detailed test results."""
        results = {
            'metrics': metrics.compute(),
            'classification_report': metrics.get_classification_report(),
            'confusion_matrix': metrics.get_confusion_matrix().tolist()
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix plot
        fig = metrics.plot_confusion_matrix()
        fig.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
