import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from model.SkateFormerPre import Model as SkateFormer
from Feeders.feeder_pretrain import FeederPretrain

def nt_xent_loss(z1, z2, temperature=0.07):
    """Symmetric NT-Xent loss for two augmented views"""
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    z = torch.cat([z1, z2], dim=0)               
    sim = torch.matmul(z, z.T) / temperature     
    
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    
    loss = F.cross_entropy(sim, labels)
    return loss

class ContrastiveTrainer:
    def __init__(self, config_path):
        self.scaler = GradScaler()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        assert self.cfg is not None, "Config YAML not loaded!"

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Using device: {self.device}')
        print(f'[INFO] GPU count: {torch.cuda.device_count()}')

    
        self.work_dir = self.cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoint_dir = self.work_dir
        print(f"[INFO] Checkpoint directory: {self.checkpoint_dir}")

        # Model
        self.model = SkateFormer(**self.cfg['model_args'])

        if torch.cuda.device_count() > 1:
            print(f'[INFO] Using {torch.cuda.device_count()} GPUs (DataParallel)')
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'[INFO] Total trainable params: {total_params / 1e6:.2f}M')

        # Optimizer
        weight_decay = self.cfg.get('weight_decay', 0.0001)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['learning_rate'],
            weight_decay=weight_decay
        )

        # Scheduler
        warmup_epochs = self.cfg.get('warm_up_epoch', 10)
        total_epochs = self.cfg['num_epoch']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.cfg.get('min_lr', 1e-5)
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = self.cfg.get('warmup_lr', 1e-4)

        # Các hyperparams khác
        self.temperature = self.cfg.get('temperature', 0.07)
        self.grad_clip = self.cfg.get('grad_clip', True)
        self.grad_max = self.cfg.get('grad_max', 1.0)
        self.early_stopping_patience = self.cfg.get('early_stopping_patience', 20)
        self.early_stopping_min_delta = self.cfg.get('early_stopping_min_delta', 0.0005)

        # State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        last_path = os.path.join(self.checkpoint_dir, "last.pth")
        if os.path.exists(last_path):
            print(f"[INFO] Found checkpoint, resuming from {last_path}")
            checkpoint = torch.load(last_path, map_location=self.device)

            # Load model
            model_state = checkpoint['model']
            self.model.load_state_dict(model_state)

            # Load optimizer & scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            # Restore state
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.early_stop_counter = 0  

            print(f"[INFO] Resume training from epoch {self.start_epoch} | Best Val Loss so far: {self.best_val_loss:.4f}")
        else:
            print("[INFO] No checkpoint found, starting from scratch")
            if self.cfg.get('weights'):
                checkpoint = torch.load(self.cfg['weights'], map_location='cpu')
                state_dict = checkpoint['model']
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded pretrained weights from {self.cfg['weights']} (strict=False)")

        # DataLoaders
        self.train_loader = self._build_dataloader('train')
        self.val_loader   = self._build_dataloader('val')

        print(f'[INFO] Optimizer: AdamW | LR: {self.cfg["learning_rate"]} | WD: {weight_decay}')
        print(f'[INFO] Warmup epochs: {warmup_epochs} | Warmup LR: {self.warmup_lr}')
        print(f'[INFO] Contrastive temperature: {self.temperature}')
        print(f'[INFO] Early stopping patience: {self.early_stopping_patience} | Delta: {self.early_stopping_min_delta}')

    def _build_dataloader(self, split):
        feeder_args = self.cfg[f'{split}_feeder_args'].copy()
        data_path = feeder_args.pop('data_path')

        dataset = FeederPretrain(
            data_path=data_path,
            split=split,
            **feeder_args
        )

        batch_size = self.cfg['batch_size'] if split == 'train' else self.cfg.get('test_batch_size', self.cfg['batch_size'])
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=self.cfg.get('num_worker', 0),
            pin_memory=True,
            drop_last=(split == 'train'),
            persistent_workers=(self.cfg.get('num_worker', 0) > 0)
        )
        return loader

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr + (self.cfg['learning_rate'] - self.warmup_lr) * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            return self.optimizer.param_groups[0]['lr']

    def train_one_epoch(self, epoch):
        self.model.train()
        current_lr = self.adjust_learning_rate(epoch)

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'[Train] Epoch {epoch:03d}')

        for batch in pbar:
            if len(batch) == 4:
                view1, view2, index_t, _ = batch
            else:
                view1, view2, index_t = batch

            view1 = view1.float().to(self.device)
            view2 = view2.float().to(self.device)
            index_t = index_t.long().to(self.device)

            self.optimizer.zero_grad()
            
            with autocast():
                z1 = self.model(view1, index_t)
                z2 = self.model(view2, index_t)
                loss = nt_xent_loss(z1, z2, self.temperature)

            self.scaler.scale(loss).backward()
            
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_max)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(loss=f'{total_loss / num_batches:.4f}', lr=f'{current_lr:.6f}')

        return total_loss / num_batches

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'[Val] Epoch {epoch:03d}'):
                if len(batch) == 4:
                    view1, view2, index_t, _ = batch
                else:
                    view1, view2, index_t = batch

                view1 = view1.float().to(self.device)
                view2 = view2.float().to(self.device)
                index_t = index_t.long().to(self.device)

                with autocast():
                    z1 = self.model(view1, index_t)
                    z2 = self.model(view2, index_t)
                    loss = nt_xent_loss(z1, z2, self.temperature)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

        state = {
            'epoch': epoch,
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        torch.save(state, os.path.join(self.checkpoint_dir, 'last.pth'))

        torch.save(state, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth'))

        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f'[INFO] Best model saved at epoch {epoch} (val_loss={val_loss:.4f})')

    def train(self):
        print('\n========== START CONTRASTIVE PRETRAIN ==========\n')

        for epoch in range(self.start_epoch, self.cfg['num_epoch']):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')

            # Early stopping logic
            is_best = val_loss < self.best_val_loss - self.early_stopping_min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            if (epoch + 1) % 10 == 0:
                print(f'{"="*60}\nSUMMARY @ Epoch {epoch}: Best Val Loss = {self.best_val_loss:.4f} | Early Stop: {self.early_stop_counter}/{self.early_stopping_patience}\n{"="*60}')

            if self.early_stop_counter >= self.early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch} | Best Val Loss: {self.best_val_loss:.4f}')
                break

        print('\n========== CONTRASTIVE PRETRAIN FINISHED ==========')


def main():
    parser = argparse.ArgumentParser(description='Contrastive Pretraining with SkateFormer')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    trainer = ContrastiveTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()