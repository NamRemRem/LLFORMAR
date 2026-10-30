import os
import sys
import time
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from model.LLFormer import LLFormer
from transform.data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="LLFormer Preliminary Training and Evaluation")
    parser.add_argument('--epochs', type=int, default=5, help='Number of preliminary epochs')
    parser.add_argument('--max_steps', type=int, default=20, help='Max batches per epoch for fast preliminary run')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--train_ps', type=int, default=128, help='Training patch size')
    parser.add_argument('--val_ps', type=int, default=128, help='Validation patch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Mixed Precision (AMP)')
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("      LLFormer Preliminary Training & Evaluation")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Device: {device} ({gpu_name})")
    print(f"Configuration: Epochs={args.epochs}, Max Steps/Epoch={args.max_steps}, Batch Size={args.batch_size}, AMP={args.use_amp}")

    # Paths
    train_dir = './LOLdataset/our485'
    val_dir = './LOLdataset/eval15'
    output_dir = './preliminary_results'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results_img_dir = os.path.join(output_dir, 'enhanced_images')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_img_dir, exist_ok=True)

    # 1. Initialize Model
    print("\n[1/4] Khởi tạo kiến trúc mô hình LLFormer...")
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

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Tổng tham số (Parameters): {param_count:,} (~{param_count / 1e6:.2f}M)")

    # 2. Data Loaders
    print("\n[2/4] Tải dữ liệu LOL Dataset...")
    train_dataset = get_training_data(train_dir, {'patch_size': args.train_ps})
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dataset = get_validation_data(val_dir, {'patch_size': args.val_ps})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"--> Số mẫu huấn luyện: {len(train_dataset)}, Số mẫu kiểm thử (Validation): {len(val_dataset)}")

    # 3. Optimizer & Scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs - 1), eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
    scheduler.step()

    # Pre-training Baseline Evaluation
    print("\n--> Đánh giá Baseline trước khi huấn luyện (Ảnh gốc thiếu sáng vs Ảnh chuẩn)...")
    baseline_psnrs = []
    baseline_ssims = []
    with torch.no_grad():
        for data_val in val_loader:
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)
            h, w = target.shape[2], target.shape[3]
            input_crop = input_[:, :, :h, :w]
            baseline_psnrs.append(utils.torchPSNR(input_crop[0], target[0]).item())
            baseline_ssims.append(utils.torchSSIM(input_crop, target).item())
    baseline_psnr = float(np.mean(baseline_psnrs))
    baseline_ssim = float(np.mean(baseline_ssims))
    print(f"    Baseline PSNR (Raw Low-Light): {baseline_psnr:.2f} dB")
    print(f"    Baseline SSIM (Raw Low-Light): {baseline_ssim:.4f}")

    # 4. Training Loop
    print(f"\n[3/4] Bắt đầu huấn luyện sơ bộ ({args.epochs} epochs, tối đa {args.max_steps} steps/epoch)...")
    history = {
        'epoch': [],
        'loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'lr': [],
        'time_per_epoch': []
    }

    best_psnr = -1.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", total=min(len(train_loader), args.max_steps))
        for step, data in enumerate(pbar):
            if step >= args.max_steps:
                break

            for p in model.parameters():
                p.grad = None
            target = data[0].to(device)
            input_ = data[1].to(device)

            with torch.amp.autocast('cuda', enabled=args.use_amp):
                restored = model(input_)
                loss = criterion(restored, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step_count += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(1, step_count)

        # Validation
        model.eval()
        epoch_psnrs = []
        epoch_ssims = []
        with torch.no_grad():
            for data_val in val_loader:
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                h, w = target.shape[2], target.shape[3]
                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    restored = model(input_)
                restored = restored[:, :, :h, :w].float()

                epoch_psnrs.append(utils.torchPSNR(restored[0], target[0]).item())
                epoch_ssims.append(utils.torchSSIM(restored, target).item())

        val_psnr = float(np.mean(epoch_psnrs))
        val_ssim = float(np.mean(epoch_ssims))
        current_lr = scheduler.get_lr()[0]
        scheduler.step()
        epoch_time = time.time() - epoch_start

        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(current_lr)
        history['time_per_epoch'].append(epoch_time)

        print(f"--> Epoch {epoch:02d}/{args.epochs:02d} | Loss: {epoch_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Time: {epoch_time:.1f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim
            }, os.path.join(checkpoint_dir, 'model_bestPSNR.pth'))

    total_time = time.time() - total_start
    print(f"\nHuấn luyện sơ bộ hoàn thành trong {total_time:.1f}s (~{total_time/60:.1f} phút)")

    # 5. Export enhanced samples & plots
    print("\n[4/4] Xuất kết quả hình ảnh và lập sơ đồ số liệu...")
    model.eval()
    sample_vis_data = []
    with torch.no_grad():
        for i, data_val in enumerate(val_loader):
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)
            filename = data_val[2][0]
            h, w = target.shape[2], target.shape[3]

            with torch.amp.autocast('cuda', enabled=args.use_amp):
                restored = model(input_)
            restored = torch.clamp(restored[:, :, :h, :w].float(), 0, 1)

            sample_psnr = utils.torchPSNR(restored[0], target[0]).item()
            sample_ssim = utils.torchSSIM(restored, target).item()

            inp_np = (input_[:, :, :h, :w][0].cpu().permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
            res_np = (restored[0].cpu().permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
            tar_np = (target[0].cpu().permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(results_img_dir, f"{filename}_enhanced.png"), cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR))

            if i < 3:
                sample_vis_data.append({
                    'name': filename,
                    'input': inp_np,
                    'restored': res_np,
                    'target': tar_np,
                    'psnr': sample_psnr,
                    'ssim': sample_ssim
                })

    # Plot 1: Metrics Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0b1120')
    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#475569')

    # Subplot 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(history['epoch'], history['loss'], marker='o', color='#38bdf8', linewidth=2.5, markersize=8, label='Training Loss (Smooth L1)')
    ax1.set_title('Đường cong mất mát (Training Loss)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    ax1.set_xlabel('Epoch', color='#cbd5e1', fontsize=11)
    ax1.set_ylabel('Loss', color='#cbd5e1', fontsize=11)
    ax1.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # Subplot 2: PSNR
    ax2 = axes[0, 1]
    ax2.axhline(y=baseline_psnr, color='#f43f5e', linestyle='--', linewidth=1.8, label=f'Baseline gốc ({baseline_psnr:.2f} dB)')
    ax2.plot(history['epoch'], history['val_psnr'], marker='s', color='#34d399', linewidth=2.5, markersize=8, label='LLFormer Val PSNR')
    ax2.set_title('Chỉ số PSNR trên tập kiểm thử (dB)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    ax2.set_xlabel('Epoch', color='#cbd5e1', fontsize=11)
    ax2.set_ylabel('PSNR (dB)', color='#cbd5e1', fontsize=11)
    ax2.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # Subplot 3: SSIM
    ax3 = axes[1, 0]
    ax3.axhline(y=baseline_ssim, color='#f43f5e', linestyle='--', linewidth=1.8, label=f'Baseline gốc ({baseline_ssim:.4f})')
    ax3.plot(history['epoch'], history['val_ssim'], marker='^', color='#fbbf24', linewidth=2.5, markersize=8, label='LLFormer Val SSIM')
    ax3.set_title('Chỉ số tương đồng cấu trúc (SSIM)', fontsize=13, fontweight='bold', color='#f8fafc', pad=10)
    ax3.set_xlabel('Epoch', color='#cbd5e1', fontsize=11)
    ax3.set_ylabel('SSIM', color='#cbd5e1', fontsize=11)
    ax3.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')

    # Subplot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['Chỉ số (Metric)', 'Ảnh gốc thiếu sáng', f'LLFormer (Epoch {args.epochs})', 'Mức cải thiện'],
        ['PSNR (dB)', f"{baseline_psnr:.2f} dB", f"{history['val_psnr'][-1]:.2f} dB", f"+{history['val_psnr'][-1] - baseline_psnr:.2f} dB"],
        ['SSIM', f"{baseline_ssim:.4f}", f"{history['val_ssim'][-1]:.4f}", f"+{history['val_ssim'][-1] - baseline_ssim:.4f}"],
        ['Loss', '-', f"{history['loss'][-1]:.4f}", f"Giảm từ {history['loss'][0]:.4f}"],
        ['Tốc độ / Epoch', '-', f"~{np.mean(history['time_per_epoch']):.1f}s", f"Tổng: {total_time:.1f}s"]
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#475569')
        if row == 0:
            cell.set_facecolor('#0284c7')
            cell.set_text_props(color='#ffffff', weight='bold')
        else:
            cell.set_facecolor('#1e293b' if row % 2 == 0 else '#334155')
            cell.set_text_props(color='#f8fafc')

    ax4.set_title('Bảng tổng hợp số liệu hiệu năng sơ bộ', fontsize=13, fontweight='bold', color='#f8fafc', pad=20)

    fig.suptitle('BÁO CÁO SƠ ĐỒ SỐ LIỆU ĐÀO TẠO & KIỂM THỬ SƠ BỘ MÔ HÌNH LLFormer', fontsize=16, fontweight='bold', color='#38bdf8', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    metrics_chart_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(metrics_chart_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"--> Đã lưu biểu đồ sơ đồ số liệu: {metrics_chart_path}")

    # Plot 2: Visual Comparison
    fig_vis, axes_vis = plt.subplots(len(sample_vis_data), 3, figsize=(15, 4 * len(sample_vis_data)))
    fig_vis.patch.set_facecolor('#0b1120')

    col_titles = ['Ảnh đầu vào (Low-Light)', 'Mô hình LLFormer khôi phục', 'Ảnh chuẩn (Ground Truth)']
    for col_idx, title in enumerate(col_titles):
        axes_vis[0, col_idx].set_title(title, fontsize=14, fontweight='bold', color='#38bdf8', pad=12)

    for row_idx, item in enumerate(sample_vis_data):
        ax_in = axes_vis[row_idx, 0]
        ax_in.imshow(item['input'])
        ax_in.set_ylabel(f"Mẫu #{item['name']}", color='#e2e8f0', fontsize=12, fontweight='bold')
        ax_in.set_xticks([]); ax_in.set_yticks([])

        ax_res = axes_vis[row_idx, 1]
        ax_res.imshow(item['restored'])
        ax_res.set_xlabel(f"PSNR: {item['psnr']:.2f} dB | SSIM: {item['ssim']:.4f}", color='#34d399', fontsize=11, fontweight='bold', labelpad=8)
        ax_res.set_xticks([]); ax_res.set_yticks([])

        ax_gt = axes_vis[row_idx, 2]
        ax_gt.imshow(item['target'])
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

    plt.tight_layout()
    vis_chart_path = os.path.join(output_dir, 'visual_comparison.png')
    plt.savefig(vis_chart_path, dpi=200, bbox_inches='tight', facecolor=fig_vis.get_facecolor())
    plt.close()
    print(f"--> Đã lưu hình ảnh so sánh trực quan: {vis_chart_path}")

    print("\n" + "=" * 65)
    print("HOÀN TẤT THÀNH CÔNG QUÁ TRÌNH CHẠY SƠ BỘ!")
    print(f"1. Biểu đồ sơ đồ số liệu: {metrics_chart_path}")
    print(f"2. Ảnh so sánh trực quan:  {vis_chart_path}")
    print(f"3. Thư mục ảnh phục hồi:  {results_img_dir}")
    print(f"4. Trọng số tốt nhất:     {os.path.join(checkpoint_dir, 'model_bestPSNR.pth')}")
    print("=" * 65)

if __name__ == '__main__':
    main()
