from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    data_root: str = "data/raw"
    processed_root: str = "data/processed"
    split_dir: str = "data/splits"
    train_split: str = "train.txt"
    val_split: str = "val.txt"
    test_split: str = "test.txt"
    img_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "Normal", "OLK", "OLP", "OSCC", "Other"
    ])


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.3
    use_attention: bool = True
    attention_type: str = "cbam"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stop_patience: int = 12
    loss: str = "cross_entropy"
    num_workers: int = 4
    use_amp: bool = True
    log_interval: int = 10
    save_dir: str = "checkpoints"
    experiment_name: str = "default"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda"
