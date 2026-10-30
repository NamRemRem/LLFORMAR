import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from model.LLFormer import LLFormer
from transform.data_RGB import get_training_data, get_validation_data
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="LLFormer 50 Epochs Training")
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--train_ps', type=int, default=128, help='Training patch size')
    parser.add_argument('--val_ps', type=int, default=128, help='Validation patch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--from_scratch', action='store_true', default=True, help='Train baseline from scratch starting from Epoch 0')
    parser.add_argument('--resume_weights', type=str, default='', help='Checkpoint weights path to resume (if any)')
    parser.add_argument('--output_dir', type=str, default='./training_50epochs_results')
    return parser.parse_args()

def save_comparison_plot(model, device, output_dir, epoch, psnr_val, ssim_val):
    from PIL import Image
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F
    import matplotlib.patches as patches
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    low_path = './LOLdataset/eval15/low/1.png'
    high_path = './LOLdataset/eval15/high/1.png'
    if not os.path.exists(low_path) or not os.path.exists(high_path):
        return

    raw_low_img = Image.open(low_path).convert('RGB')
    raw_high_img = Image.open(high_path).convert('RGB')

    low_tensor = TF.to_tensor(raw_low_img).unsqueeze(0).to(device)
    h, w = low_tensor.shape[2], low_tensor.shape[3]
    mul = 16
    H, W = ((h + mul - 1) // mul) * mul, ((w + mul - 1) // mul) * mul
    padh, padw = H - h, W - w
    inp_padded = F.pad(low_tensor, (0, padw, 0, padh), 'reflect')

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            restored = model(inp_padded)
        restored = torch.clamp(restored[:, :, :h, :w].float(), 0, 1)

    img_low = np.array(raw_low_img)
    img_gt = np.array(raw_high_img)
    img_pred = (restored[0].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    metric_psnr_in = psnr(img_gt, img_low)
    metric_ssim_in = ssim(img_gt, img_low, channel_axis=2)
    metric_psnr_pred = psnr(img_gt, img_pred)
    metric_ssim_pred = ssim(img_gt, img_pred, channel_axis=2)

    box_size = 90
    box_y = int(h * 0.45)
    box_x = int(w * 0.40)
    crop_low = img_low[box_y:box_y+box_size, box_x:box_x+box_size]
    crop_pred = img_pred[box_y:box_y+box_size, box_x:box_x+box_size]
    crop_gt = img_gt[box_y:box_y+box_size, box_x:box_x+box_size]
    diff_map = np.mean(np.abs(img_gt.astype(float) - img_pred.astype(float)), axis=2)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12), facecolor='#090d16')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_low)
    ax1.set_title(f"[1] Low-Light Input (Gốc)\nPSNR: {metric_psnr_in:.2f} dB | SSIM: {metric_ssim_in:.4f}", color='#f43f5e', fontsize=12, fontweight='bold', pad=8)
    ax1.add_patch(patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#ef4444', facecolor='none', linestyle='--'))
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_pred)
    ax2.set_title(f"[2] LLFormer (Epoch {epoch})\nPSNR: {metric_psnr_pred:.2f} dB | SSIM: {metric_ssim_pred:.4f}", color='#34d399', fontsize=12, fontweight='bold', pad=8)
    ax2.add_patch(patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#10b981', facecolor='none', linestyle='--'))
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_gt)
    ax3.set_title(f"[3] Ground Truth (Ảnh đối chứng)\nTarget Reference", color='#38bdf8', fontsize=12, fontweight='bold', pad=8)
    ax3.add_patch(patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#0284c7', facecolor='none', linestyle='--'))
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 0]); ax4.imshow(crop_low); ax4.set_title("Vùng phóng to: Low-Light", color='#fda4af', fontsize=11); ax4.axis('off')
    ax5 = fig.add_subplot(gs[1, 1]); ax5.imshow(crop_pred); ax5.set_title(f"Vùng phóng to: LLFormer (Epoch {epoch})", color='#6ee7b7', fontsize=11); ax5.axis('off')
    ax6 = fig.add_subplot(gs[1, 2]); ax6.imshow(crop_gt); ax6.set_title("Vùng phóng to: Ground Truth", color='#7dd3fc', fontsize=11); ax6.axis('off')

    ax7 = fig.add_subplot(gs[2, 0])
    im_err = ax7.imshow(diff_map, cmap='inferno', vmin=0, vmax=100)
    ax7.set_title("Bản đồ sai số tuyệt đối |LLFormer - GT|", color='#e2e8f0', fontsize=11, fontweight='bold', pad=6)
    ax7.axis('off')
    cbar = fig.colorbar(im_err, ax=ax7, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#e2e8f0', labelsize=9)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_facecolor('#131b2e')
    gray_low = np.mean(img_low, axis=2).ravel()
    gray_pred = np.mean(img_pred, axis=2).ravel()
    gray_gt = np.mean(img_gt, axis=2).ravel()
    ax8.hist(gray_low, bins=50, color='#f43f5e', alpha=0.55, label='Low-Light', density=True)
    ax8.hist(gray_pred, bins=50, color='#34d399', alpha=0.55, label=f'LLFormer Ep{epoch}', density=True)
    ax8.hist(gray_gt, bins=50, color='#38bdf8', alpha=0.45, histtype='step', linewidth=2, label='Ground Truth', density=True)
    ax8.set_title("Biểu đồ phân bố độ sáng (Histogram)", color='#e2e8f0', fontsize=11, fontweight='bold', pad=8)
    ax8.set_xlabel("Mức độ sáng (0 - 255)", color='#94a3b8', fontsize=9)
    ax8.tick_params(colors='#94a3b8', labelsize=8)
    ax8.legend(fontsize=8, facecolor='#0b1120', edgecolor='#334155', labelcolor='#e2e8f0')

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    table_vals = [
        ['Chỉ số đánh giá', 'Ảnh Low-Light', f'LLFormer Ep {epoch}', 'Cải thiện'],
        ['PSNR (dB)', f"{metric_psnr_in:.2f} dB", f"{metric_psnr_pred:.2f} dB", f"+{metric_psnr_pred - metric_psnr_in:.2f} dB"],
        ['SSIM (Cấu trúc)', f"{metric_ssim_in:.4f}", f"{metric_ssim_pred:.4f}", f"+{metric_ssim_pred - metric_ssim_in:.4f}"],
        ['Độ phân giải', f"{w} x {h}", f"{w} x {h}", "Full-Resolution"],
        ['Trạng thái', 'Chưa xử lý', f'Đạt Epoch {epoch}/50', 'Tối ưu liên tục']
    ]
    tbl = ax9.table(cellText=table_vals, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.05, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#334155')
        if r == 0: cell.set_facecolor('#0284c7'); cell.set_text_props(color='#ffffff', weight='bold')
        else: cell.set_facecolor('#131b2e' if r % 2 == 0 else '#1e293b'); cell.set_text_props(color='#f1f5f9')
    ax9.set_title(f"Bảng tiến trình phục hồi (Epoch {epoch})", color='#e2e8f0', fontsize=11, fontweight='bold', pad=15)

    fig.suptitle(f"TIẾN TRÌNH KHÔI PHỤC LLFORMER: EPOCH {epoch}/50 (Ảnh mẫu: 1.png)", fontsize=15, fontweight='bold', color='#38bdf8', y=0.98)
    plt.savefig(os.path.join(output_dir, f'comparison_epoch_{epoch}.png'), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(os.path.join(output_dir, 'comparison_latest.png'), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def plot_curves(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0b1120')
    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')
        for spine in ax.spines.values(): spine.set_color('#475569')

    # Loss
    axes[0, 0].plot(history['epoch'], history['loss'], color='#38bdf8', linewidth=2.5, marker='o', markersize=4, label='Training Loss')
    axes[0, 0].set_title('Đường cong mất mát (Training Loss)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    axes[0, 0].set_xlabel('Epoch', color='#cbd5e1'); axes[0, 0].set_ylabel('Loss', color='#cbd5e1')
    axes[0, 0].legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # PSNR
    axes[0, 1].plot(history['epoch'], history['val_psnr'], color='#34d399', linewidth=2.5, marker='s', markersize=4, label='Validation PSNR (dB)')
    axes[0, 1].set_title('Chỉ số PSNR trên tập kiểm thử (dB)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    axes[0, 1].set_xlabel('Epoch', color='#cbd5e1'); axes[0, 1].set_ylabel('PSNR (dB)', color='#cbd5e1')
    axes[0, 1].legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # SSIM
    axes[1, 0].plot(history['epoch'], history['val_ssim'], color='#fbbf24', linewidth=2.5, marker='^', markersize=4, label='Validation SSIM')
    axes[1, 0].set_title('Độ tương đồng cấu trúc (SSIM)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    axes[1, 0].set_xlabel('Epoch', color='#cbd5e1'); axes[1, 0].set_ylabel('SSIM', color='#cbd5e1')
    axes[1, 0].legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # Learning Rate
    axes[1, 1].plot(history['epoch'], history['lr'], color='#a78bfa', linewidth=2, label='Learning Rate')
    axes[1, 1].set_title('Tốc độ học (Learning Rate Schedule)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    axes[1, 1].set_xlabel('Epoch', color='#cbd5e1'); axes[1, 1].set_ylabel('Learning Rate', color='#cbd5e1')
    axes[1, 1].legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    fig.suptitle('TIẾN TRÌNH HUẤN LUYỆN LLFORMER (50 EPOCHS)', fontsize=15, fontweight='bold', color='#38bdf8', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def main():
    args = parse_args()

    print("=" * 65)
    print("      LLFormer 50-Epoch Extended Training Pipeline")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Model
    print("\n[1/4] Khởi tạo kiến trúc LLFormer và nạp trọng số tiếp tục...")
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

    if not args.from_scratch and args.resume_weights and os.path.exists(args.resume_weights):
        ckpt = torch.load(args.resume_weights, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        prev_epoch = ckpt.get('epoch', 0)
        print(f"--> Đã nạp thành công checkpoint từ '{args.resume_weights}' (Epoch {prev_epoch})!")
    else:
        print("--> Khởi tạo ngẫu nhiên mô hình LLFormer (Huấn luyện baseline từ đầu: Epoch 0).")

    # 2. Dataloaders (Tải toàn bộ tập dữ liệu LOL)
    train_dir = './LOLdataset/our485'
    val_dir = './LOLdataset/eval15'
    print(f"\n[2/4] Tải toàn bộ tập dữ liệu huấn luyện {train_dir} (485 cặp ảnh)...")
    train_dataset = get_training_data(train_dir, {'patch_size': args.train_ps})
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = get_validation_data(val_dir, {'patch_size': args.val_ps})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"--> Số batch mỗi epoch: {len(train_loader)} (Batch size: {args.batch_size})")

    # 3. Optimizer & Cosine Annealing
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    history = {'epoch': [], 'loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': [], 'time': []}
    best_psnr = -1.0
    start_time_all = time.time()

    print(f"\n[3/4] Bắt đầu huấn luyện {args.epochs} epochs từ Epoch 0...")
    print(f"--> Cập nhật biểu đồ và ảnh so sánh trực quan định kỳ!")

    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs-1:02d}")
        for data in pbar:
            for p in model.parameters(): p.grad = None
            target = data[0].to(device)
            input_ = data[1].to(device)

            with torch.amp.autocast('cuda'):
                restored = model(input_)
                loss = criterion(restored, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        epoch_psnrs, epoch_ssims = [], []
        with torch.no_grad():
            for data_val in val_loader:
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                h, w = target.shape[2], target.shape[3]
                with torch.amp.autocast('cuda'):
                    restored = model(input_)
                restored = restored[:, :, :h, :w].float()
                epoch_psnrs.append(utils.torchPSNR(restored[0], target[0]).item())
                epoch_ssims.append(utils.torchSSIM(restored, target).item())

        val_psnr = float(np.mean(epoch_psnrs))
        val_ssim = float(np.mean(epoch_ssims))
        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.lr
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(cur_lr)
        history['time'].append(epoch_time)

        print(f"--> [Epoch {epoch:02d}/{args.epochs-1:02d}] Loss: {epoch_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | LR: {cur_lr:.6f} | Thời gian: {epoch_time:.1f}s")

        # Lưu best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim
            }, os.path.join(checkpoint_dir, 'model_bestPSNR.pth'))
            print(f"    *** Đã lưu Checkpoint Tốt Nhất Mới: {val_psnr:.2f} dB (SSIM: {val_ssim:.4f})! ***")

        # Lưu checkpoint mới nhất
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': val_psnr,
            'ssim': val_ssim
        }, os.path.join(checkpoint_dir, 'model_latest.pth'))

        # Cập nhật đường cong và biểu đồ định kỳ
        plot_curves(history, args.output_dir)

        # Xuất biểu đồ so sánh ảnh trực quan ở Epoch 0, 5, 10, 20, 30, 40, args.epochs-1
        if epoch in [0, 5, 10, 20, 30, 40, args.epochs - 1]:
            print(f"--> Đang cập nhật biểu đồ đối chiếu Matplotlib cho Epoch {epoch}...")
            save_comparison_plot(model, device, args.output_dir, epoch, val_psnr, val_ssim)

        # Lưu log json
        with open(os.path.join(args.output_dir, 'training_progress.json'), 'w') as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time_all
    print("\n" + "=" * 65)
    print(f"HOÀN TẤT HUẤN LUYỆN 50 EPOCHS trong {total_time/60:.1f} phút!")
    print(f"Best PSNR đạt được: {best_psnr:.2f} dB")
    print(f"Checkpoint tốt nhất lưu tại: {os.path.join(checkpoint_dir, 'model_bestPSNR.pth')}")
    print("=" * 65)

if __name__ == '__main__':
    main()
