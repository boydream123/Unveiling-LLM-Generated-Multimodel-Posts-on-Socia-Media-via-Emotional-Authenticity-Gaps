"""
Configuration management for EAGLE framework.
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    hidden_dim: int = 4096
    num_emotions: int = 7
    dropout: float = 0.1
    freeze_backbone: bool = False
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    lambda_emotion: float = 0.1
    

@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    image_size: int = 224
    max_text_length: int = 512
    num_workers: int = 4
    

@dataclass
class EAGLEConfig:
    """Main EAGLE configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    output_dir: str = "./outputs"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    seed: int = 42
    device: str = "cuda"
    use_wandb: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EAGLEConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'output_dir': self.output_dir,
            'logging_steps': self.logging_steps,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'seed': self.seed,
            'device': self.device,
            'use_wandb': self.use_wandb,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
