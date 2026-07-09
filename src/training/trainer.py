import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.scaler = GradScaler(enabled=config.training.use_amp)
        self.writer = SummaryWriter(
            log_dir=os.path.join("logs", config.training.experiment_name)
        )
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({
                "Loss": f"{total_loss / (batch_idx + 1):.4f}",
                "Acc": f"{100. * correct / total:.2f}%",
            })
        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss / len(self.val_loader), 100. * correct / total

    def fit(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        print(f"Training for {num_epochs} epochs...")
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", lr, epoch)
            if self.scheduler:
                self.scheduler.step()
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {lr:.2e} | {elapsed:.0f}s")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            self._save_checkpoint(epoch, val_acc)
        print(f"Training completed. Best val accuracy: {self.best_val_acc:.2f}%")
        return self.history

    def _save_checkpoint(self, epoch, accuracy, is_best=False):
        save_dir = self.config.training.save_dir
        exp_dir = os.path.join(save_dir, self.config.training.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        path = os.path.join(exp_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "config": self.config,
        }, path)
