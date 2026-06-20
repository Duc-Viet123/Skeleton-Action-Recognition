import os
import sys
import yaml
import argparse
import importlib
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def import_class(name):
    """Import động class từ string 'module.ClassName'."""
    parts = name.rsplit('.', 1)
    module = importlib.import_module(parts[0])
    return getattr(module, parts[1])


class STGCNTrainer:
    def __init__(self, cfg_path):
        #Load config
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self._resolve_config_paths()
        seed = self.cfg.get('seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(f"[INFO] Seed = {seed}")

        #Device
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.work_dir = self.cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'plots'), exist_ok=True)

        #AMP
        self.use_amp = bool(self.cfg.get('use_amp', True) and self.device.type == 'cuda')
        self.scaler  = GradScaler(self.device.type, enabled=self.use_amp)

        #Model
        ModelClass = import_class(self.cfg['model'])
        self.model = ModelClass(**self.cfg['model_args']).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Model: {self.cfg['model']}")
        print(f"[INFO] Params: {total_params:,}")

        #Dataset
        FeederClass = import_class(self.cfg.get('feeder', 'Feeders.feeder_finetune.FeederMultiClass'))
        train_set = FeederClass(**self.cfg['train_feeder_args'])
        val_set   = FeederClass(**self.cfg['val_feeder_args'])
        test_set  = FeederClass(**self.cfg['test_feeder_args'])

        if len(train_set) < self.cfg['batch_size']:
            raise ValueError(
                f"Train set has {len(train_set)} samples, smaller than batch_size={self.cfg['batch_size']}."
            )

        self.train_loader = DataLoader(
            train_set, batch_size=self.cfg['batch_size'],
            shuffle=True, num_workers=self.cfg['num_worker'],
            pin_memory=self.device.type == 'cuda', drop_last=True
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.cfg['batch_size'],
            shuffle=False, num_workers=self.cfg['num_worker'],
            pin_memory=self.device.type == 'cuda'
        )
        self.test_loader = DataLoader(
            test_set, batch_size=self.cfg['batch_size'],
            shuffle=False, num_workers=self.cfg['num_worker'],
            pin_memory=self.device.type == 'cuda'
        )
        print(f"[INFO] Dataset -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

        #Optimizer
        base_lr = self.cfg['lr']
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'fcn' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': base_lr * 0.1},
            {'params': head_params,     'lr': base_lr},
        ], weight_decay=self.cfg.get('weight_decay', 0.01))

        print(f"[INFO] Backbone lr={base_lr*0.1:.6f} | Head(fcn) lr={base_lr:.6f}")
        warmup_epochs = self.cfg.get('warmup_epoch', 10)
        total_epochs  = self.cfg['num_epoch']
        pct_start = min(max(warmup_epochs / total_epochs, 1e-6), 0.95)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[base_lr * 0.1, base_lr],
            total_steps=total_epochs * len(self.train_loader),
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1000.0,
        )
        # Loss
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.cfg.get('label_smoothing', 0.1)
        )

        self.train_losses, self.val_losses = [], []
        self.train_accs,   self.val_accs   = [], []
        self.best_val_acc = -1.0
        self.best_epoch   = 0
        self.no_improve   = 0
        self.patience     = self.cfg.get('early_stopping_patience', 30)
        self.class_names  = self.cfg.get('classes', ['Normal', 'Fall', 'Fight'])

        print("\n" + "=" * 60)
        print("  ST-GCN BENCHMARK TRAINING")
        print("=" * 60 + "\n")

        # Sanity check
        print("[SANITY CHECK] Validating before training...")
        init_val = self.validate(self.val_loader)
        print(f"Initial Val Acc: {init_val['accuracy']:.2f}%\n")

    def _resolve_config_paths(self):
        self.cfg['work_dir'] = self._resolve_path(self.cfg['work_dir'])
        for args_key in ('train_feeder_args', 'val_feeder_args', 'test_feeder_args'):
            feeder_args = self.cfg.get(args_key, {})
            for path_key in ('data_path', 'label_path'):
                if path_key in feeder_args:
                    feeder_args[path_key] = self._resolve_path(feeder_args[path_key])

    @staticmethod
    def _resolve_path(path):
        if not path or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(ROOT, path))

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for data_dict, label in loader:
            x     = data_dict['joint'].to(self.device)  # (N, C, T, V, M)
            label = label.to(self.device)

            with autocast(self.device.type, enabled=self.use_amp):
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, label)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds) * 100
        f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

        return {
            'loss':     total_loss / len(loader),
            'accuracy': acc,
            'f1_macro': f1,
            'preds':    all_preds,
            'labels':   all_labels,
        }

    #Train

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total   = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [TRAIN]")

        for data_dict, label in pbar:
            x     = data_dict['joint'].to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp):
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, label)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            pred     = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total   += label.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc':  f'{100*correct/total:.2f}%',
            })

        return total_loss / len(self.train_loader), 100.0 * correct / total


    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc':         self.best_val_acc,
            'config':               self.cfg,
        }
        torch.save(state, os.path.join(self.work_dir, 'last.pt'))
        if is_best:
            torch.save(state, os.path.join(self.work_dir, 'best.pt'))
            print(f"  -> NEW BEST: Val Acc {metrics['accuracy']:.2f}%")

    #Main

    def train(self):
        for epoch in range(1, self.cfg['num_epoch'] + 1):
            tr_loss, tr_acc = self.train_one_epoch(epoch)
            val_metrics     = self.validate(self.val_loader)

            self.train_losses.append(tr_loss)
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(tr_acc)
            self.val_accs.append(val_metrics['accuracy'])

            print(
                f"\nEpoch {epoch:03d} | "
                f"Train: {tr_acc:.2f}% | "
                f"Val: {val_metrics['accuracy']:.2f}% | "
                f"F1: {val_metrics['f1_macro']:.2f}%"
            )

            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch   = epoch
                self.no_improve   = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement after {self.patience} epochs)")
                break

        epochs = range(1, len(self.train_accs) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses,   label='Val Loss')
        plt.legend(); plt.grid(); plt.title('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, label='Train Acc')
        plt.plot(epochs, self.val_accs,   label='Val Acc')
        plt.axvline(self.best_epoch, color='red', linestyle='--',
                    label=f'Best (E{self.best_epoch})')
        plt.legend(); plt.grid(); plt.title('Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'plots', 'curves.png'), dpi=200)
        plt.close()
        print("\n" + "=" * 60)
        print("  TESTING BEST MODEL")
        print("=" * 60)
        best_ckpt = torch.load(os.path.join(self.work_dir, 'best.pt'), map_location=self.device)
        self.model.load_state_dict(best_ckpt['model_state_dict'])
        test_metrics = self.validate(self.test_loader)

        print(f"\n  FINAL TEST RESULTS:")
        print(f"    Accuracy : {test_metrics['accuracy']:.2f}%")
        print(f"    F1-macro : {test_metrics['f1_macro']:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Test Confusion Matrix\nAccuracy: {test_metrics["accuracy"]:.2f}%')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'plots', 'confusion_matrix.png'), dpi=200)
        plt.close()

        # Classification report
        report = classification_report(
            test_metrics['labels'], test_metrics['preds'],
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )
        with open(os.path.join(self.work_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Best Val Acc : {self.best_val_acc:.2f}% (Epoch {self.best_epoch})\n")
            f.write(f"Test Acc     : {test_metrics['accuracy']:.2f}%\n")
            f.write(f"Test F1      : {test_metrics['f1_macro']:.2f}%\n\n")
            f.write(report)
        print("\n" + report)

        print(f"\n{'='*60}")
        print("  DONE!")
        print(f"    Best Val : {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"    Test Acc : {test_metrics['accuracy']:.2f}%")
        print(f"    Test F1  : {test_metrics['f1_macro']:.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN benchmark')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to yaml config file')
    args = parser.parse_args()

    trainer = STGCNTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
