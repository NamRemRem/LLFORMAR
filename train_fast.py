import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import time
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.LLFormer import LLFormer
import utils

# Bật cuDNN benchmark tối ưu hóa phần cứng
torch.backends.cudnn.benchmark = True

class FastRAMDataset(Dataset):
    """Tải toàn bộ 485 ảnh vào RAM 1 lần duy nhất, loại bỏ hoàn toàn nghẽn cổ chai ổ đĩa"""
    def __init__(self, rgb_dir, patch_size=128):
        super().__init__()
        self.ps = patch_size
        low_dir = os.path.join(rgb_dir, 'low')
        high_dir = os.path.join(rgb_dir, 'high')

        inp_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        tar_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        print(f"--> Đang nạp nhanh {len(inp_files)} ảnh vào RAM (chỉ mất vài giây)...")
        self.samples = []
        t0 = time.time()
        for f_low, f_high in zip(inp_files, tar_files):
            img_l = TF.to_tensor(Image.open(os.path.join(low_dir, f_low)).convert('RGB'))
            img_h = TF.to_tensor(Image.open(os.path.join(high_dir, f_high)).convert('RGB'))
            self.samples.append((img_h, img_l))
        print(f"--> Đã nạp xong toàn bộ dataset vào RAM trong {time.time()-t0:.1f}s! (Không còn độ trễ đọc ổ đĩa)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_img, inp_img = self.samples[idx]
        ps = self.ps
        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        crop_tar = tar_img[:, rr:rr + ps, cc:cc + ps]
        crop_inp = inp_img[:, rr:rr + ps, cc:cc + ps]

        # Augmentation nhanh
        aug = random.randint(0, 4)
        if aug == 1:
            crop_inp = crop_inp.flip(1); crop_tar = crop_tar.flip(1)
        elif aug == 2:
            crop_inp = crop_inp.flip(2); crop_tar = crop_tar.flip(2)
        elif aug == 3:
            crop_inp = torch.rot90(crop_inp, 1, [1, 2]); crop_tar = torch.rot90(crop_tar, 1, [1, 2])

        return crop_tar, crop_inp

def parse_args():
    parser = argparse.ArgumentParser(description="LLFormer Fast Turbo Training")
    parser.add_argument('--start_epoch', type=int, default=None, help='Epoch bat dau resume (mac dinh tu dong)')
    parser.add_argument('--total_epochs', type=int, default=50, help='Tong so epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.2e-4, help='Learning rate')
    parser.add_argument('--weights', type=str, default=None, help='Duong dan checkpoint')
    parser.add_argument('--output_dir', type=str, default='./training_50epochs_results')
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("      LLFormer Turbo Training (RAM-Cached, Zero-Bottleneck)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Đọc lại history nếu có
    history_file = os.path.join(args.output_dir, 'training_progress.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = {'epoch': [], 'loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': [], 'time': []}
    else:
        history = {'epoch': [], 'loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': [], 'time': []}

    # Tự động chọn epoch bắt đầu và checkpoint
    if args.start_epoch is None:
        if history['epoch']:
            args.start_epoch = history['epoch'][-1] + 1
        else:
            args.start_epoch = 1

    if args.weights is None:
        latest_ckpt = os.path.join(checkpoint_dir, 'model_latest.pth')
        best_ckpt = os.path.join(checkpoint_dir, 'model_bestPSNR.pth')
        if os.path.exists(latest_ckpt):
            args.weights = latest_ckpt
        elif os.path.exists(best_ckpt):
            args.weights = best_ckpt

    best_psnr = max(history['val_psnr']) if history['val_psnr'] else -1.0
    best_ssim = max(history['val_ssim']) if history['val_ssim'] else -1.0

    # 1. Khởi tạo mô hình
    model = LLFormer(
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=[2, 4, 8, 16],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        attention=True,
        skip=False
    ).to(device)

    # Resume checkpoint
    ckpt_data = None
    if args.weights and os.path.exists(args.weights):
        ckpt_data = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt_data['state_dict'])
        ckpt_epoch = ckpt_data.get('epoch', args.start_epoch - 1)
        print(f"--> Nạp checkpoint thành công: {args.weights} (Epoch {ckpt_epoch})")
        print(f"    Best PSNR hiện tại: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    else:
        print("--> Khởi tạo mô hình từ đầu.")

    # 2. Dataloader RAM siêu tốc
    train_dataset = FastRAMDataset('./LOLdataset/our485', patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    # Val dataset
    from transform.data_RGB import get_validation_data
    val_dataset = get_validation_data('./LOLdataset/eval15', {'patch_size': 128})
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # 3. Optimizer & Scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    if ckpt_data and 'optimizer' in ckpt_data:
        try:
            optimizer.load_state_dict(ckpt_data['optimizer'])
        except Exception:
            pass

    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs, eta_min=1e-6)

    print(f"\n--> Bắt đầu chạy Turbo từ Epoch {args.start_epoch} đến Epoch {args.total_epochs}...")

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.total_epochs:02d}")
        for target, input_ in pbar:
            for p in model.parameters(): p.grad = None
            target = target.to(device, non_blocking=True)
            input_ = input_.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                restored = model(input_)
                loss = criterion(restored, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)

        # Validation siêu tốc
        model.eval()
        epoch_psnrs, epoch_ssims = [], []
        with torch.no_grad():
            for data_val in val_loader:
                tar = data_val[0].to(device, non_blocking=True)
                inp = data_val[1].to(device, non_blocking=True)
                h, w = tar.shape[2], tar.shape[3]
                with torch.amp.autocast('cuda'):
                    res = model(inp)
                res = res[:, :, :h, :w].float()
                epoch_psnrs.append(utils.torchPSNR(res[0], tar[0]).item())
                epoch_ssims.append(utils.torchSSIM(res, tar).item())

        val_psnr = float(np.mean(epoch_psnrs))
        val_ssim = float(np.mean(epoch_ssims))
        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Giải phóng cache CUDA tránh phân mảnh VRAM
        torch.cuda.empty_cache()

        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(cur_lr)
        history['time'].append(epoch_time)

        print(f"--> [Epoch {epoch:02d}/{args.total_epochs:02d}] Loss: {epoch_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Thời gian: {epoch_time:.1f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim
            }, os.path.join(checkpoint_dir, 'model_bestPSNR.pth'))
            print(f"    *** KỶ LỤC MỚI: PSNR = {val_psnr:.2f} dB | SSIM = {val_ssim:.4f} ***")

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': val_psnr,
            'ssim': val_ssim
        }, os.path.join(checkpoint_dir, 'model_latest.pth'))

        # Lưu json
        with open(os.path.join(args.output_dir, 'training_progress.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 65)
    print(f"HOÀN TẤT HUẤN LUYỆN TURBO! Best PSNR: {best_psnr:.2f} dB")
    print("=" * 65)

if __name__ == '__main__':
    main()
