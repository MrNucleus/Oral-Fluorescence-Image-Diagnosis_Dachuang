import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import OralFluorescenceDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.backbone import create_model
from src.training.trainer import Trainer
from src.training.loss import get_loss_fn
from src.utils.seed import set_seed


def main():
    config = Config()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default=config.model.backbone)
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size)
    parser.add_argument("--epochs", type=int, default=config.training.num_epochs)
    parser.add_argument("--lr", type=float, default=config.training.learning_rate)
    parser.add_argument("--loss", type=str, default=config.training.loss)
    parser.add_argument("--attention", action="store_true", default=config.model.use_attention)
    parser.add_argument("--no-attention", action="store_false", dest="attention")
    parser.add_argument("--attention-type", type=str, default=config.model.attention_type)
    parser.add_argument("--exp-name", type=str, default=config.training.experiment_name)
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    config.model.backbone = args.backbone
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.loss = args.loss
    config.model.use_attention = args.attention
    config.model.attention_type = args.attention_type
    config.training.experiment_name = args.exp_name
    config.seed = args.seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Device: {device}")

    train_transform = get_train_transforms(config.data.img_size)
    val_transform = get_val_transforms(config.data.img_size)

    train_dataset = OralFluorescenceDataset(
        config.data.data_root,
        os.path.join(config.data.split_dir, config.data.train_split),
        transform=train_transform,
        class_names=config.data.class_names,
    )
    val_dataset = OralFluorescenceDataset(
        config.data.data_root,
        os.path.join(config.data.split_dir, config.data.val_split),
        transform=val_transform,
        class_names=config.data.class_names,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = create_model(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.backbone} | Params: {params:,}")

    criterion = get_loss_fn(config.training.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)

    trainer = Trainer(model, train_loader, val_loader, criterion,
                      optimizer, scheduler, config, device)
    trainer.fit()
    print(f"Best val accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
