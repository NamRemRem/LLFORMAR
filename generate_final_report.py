import os
import sys
import json
import time
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from model.LLFormer import LLFormer
import utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"
RESULTS_DIR = r"./training_50epochs_results"
CKPT_PATH = os.path.join(RESULTS_DIR, "checkpoints", "model_bestPSNR.pth")
PROGRESS_JSON = os.path.join(RESULTS_DIR, "training_progress.json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def plot_training_curves():
    print("--> Đang vẽ biểu đồ huấn luyện 38 epochs...")
    with open(PROGRESS_JSON, 'r') as f:
        data = json.load(f)

    epochs = data['epoch']
    losses = data['loss']
    psnrs = data['val_psnr']
    ssims = data['val_ssim']
    lrs = data['lr']

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor='#ffffff')
    for ax in axes.flat:
        ax.set_facecolor('#f7f9fc')
        ax.tick_params(colors='#333333', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
        ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

    # 1. Loss
    axes[0, 0].plot(epochs, losses, color='#d62728', linewidth=2.5, marker='o', markersize=4, label='Smooth L1 Loss')
    axes[0, 0].set_title('Training Loss (Hội tụ qua 38 Epochs)', color='#1a1a1a', fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Epoch', color='#444444', fontsize=11)
    axes[0, 0].set_ylabel('Loss', color='#444444', fontsize=11)
    axes[0, 0].legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')

    # 2. PSNR
    best_psnr_idx = int(np.argmax(psnrs))
    axes[0, 1].plot(epochs, psnrs, color='#1f77b4', linewidth=2.5, marker='s', markersize=4, label='Validation PSNR')
    axes[0, 1].scatter([epochs[best_psnr_idx]], [psnrs[best_psnr_idx]], color='#e6a817', s=130, zorder=5, label=f'Best PSNR: {psnrs[best_psnr_idx]:.2f} dB (Ep {epochs[best_psnr_idx]})')
    axes[0, 1].set_title('Validation PSNR (Độ rõ nét & Khử nhiễu)', color='#1a1a1a', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Epoch', color='#444444', fontsize=11)
    axes[0, 1].set_ylabel('PSNR (dB)', color='#444444', fontsize=11)
    axes[0, 1].legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')

    # 3. SSIM
    best_ssim_idx = int(np.argmax(ssims))
    axes[1, 0].plot(epochs, ssims, color='#2ca02c', linewidth=2.5, marker='^', markersize=4, label='Validation SSIM')
    axes[1, 0].scatter([epochs[best_ssim_idx]], [ssims[best_ssim_idx]], color='#e6a817', s=130, zorder=5, label=f'Best SSIM: {ssims[best_ssim_idx]:.4f} (Ep {epochs[best_ssim_idx]})')
    axes[1, 0].set_title('Validation SSIM (Độ tương đồng cấu trúc)', color='#1a1a1a', fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Epoch', color='#444444', fontsize=11)
    axes[1, 0].set_ylabel('SSIM', color='#444444', fontsize=11)
    axes[1, 0].legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, lrs, color='#8c5a00', linewidth=2.2, linestyle='-', label='Cosine Annealing LR')
    axes[1, 1].set_title('Learning Rate Schedule', color='#1a1a1a', fontsize=13, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Epoch', color='#444444', fontsize=11)
    axes[1, 1].set_ylabel('LR', color='#444444', fontsize=11)
    axes[1, 1].legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333')

    plt.suptitle('LLFormer - Quá trình huấn luyện 38 Epochs trên LOL Dataset', color='#1a1a1a', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_local = os.path.join(RESULTS_DIR, "final_training_curves.png")
    out_artifact = os.path.join(ARTIFACTS_DIR, "final_training_curves.png")
    plt.savefig(out_local, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.savefig(out_artifact, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"--> Đã lưu biểu đồ: {out_artifact}")

def evaluate_and_visualize(samples=['1.png', '22.png', '669.png', '778.png']):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Đang nạp mô hình checkpoint kỷ lục ({CKPT_PATH}) lên {device}...")

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

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    eval_results = []
    eval15_dir = './LOLdataset/eval15'
    all_files = sorted([f for f in os.listdir(os.path.join(eval15_dir, 'low')) if f.endswith(('.png', '.jpg'))])

    print(f"--> Đang đánh giá toàn bộ {len(all_files)} ảnh trong LOL eval15...")
    all_psnrs, all_ssims = [], []

    with torch.no_grad():
        for fname in all_files:
            low_p = os.path.join(eval15_dir, 'low', fname)
            high_p = os.path.join(eval15_dir, 'high', fname)
            raw_low = Image.open(low_p).convert('RGB')
            raw_high = Image.open(high_p).convert('RGB')

            t_low = TF.to_tensor(raw_low).unsqueeze(0).to(device)
            h, w = t_low.shape[2], t_low.shape[3]
            mul = 16
            H, W = ((h + mul - 1) // mul) * mul, ((w + mul - 1) // mul) * mul
            pad_low = F.pad(t_low, (0, W - w, 0, H - h), 'reflect')

            pred = model(pad_low)
            pred = torch.clamp(pred[:, :, :h, :w], 0, 1)

            img_low = np.array(raw_low)
            img_gt = np.array(raw_high)
            img_pred = (pred[0].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            p_val = psnr(img_gt, img_pred)
            s_val = ssim(img_gt, img_pred, channel_axis=2)
            all_psnrs.append(p_val)
            all_ssims.append(s_val)

            if fname in samples:
                eval_results.append({
                    'fname': fname,
                    'raw_low': raw_low,
                    'raw_high': raw_high,
                    'img_low': img_low,
                    'img_gt': img_gt,
                    'img_pred': img_pred,
                    'psnr': p_val,
                    'ssim': s_val,
                    'psnr_raw': psnr(img_gt, img_low),
                    'ssim_raw': ssim(img_gt, img_low, channel_axis=2)
                })

    mean_psnr = float(np.mean(all_psnrs))
    mean_ssim = float(np.mean(all_ssims))
    print(f"--> [LOL eval15 Full Test] Mean PSNR: {mean_psnr:.2f} dB | Mean SSIM: {mean_ssim:.4f}")

    # Tạo ảnh so sánh trực quan đa khung hình (4-sample gallery)
    fig, axes = plt.subplots(len(eval_results), 3, figsize=(16, 4.8 * len(eval_results)), facecolor='#ffffff')
    if len(eval_results) == 1:
        axes = np.expand_dims(axes, 0)

    for i, res in enumerate(eval_results):
        fn = res['fname']
        h, w = res['img_low'].shape[:2]
        # ROI box
        bs = int(min(h, w) * 0.28)
        by, bx = int(h * 0.38), int(w * 0.36)

        # 1. Low-light
        ax0 = axes[i, 0]
        ax0.imshow(res['img_low'])
        ax0.set_title(f"Ảnh tối gốc ({fn})\nPSNR: {res['psnr_raw']:.2f} dB | SSIM: {res['ssim_raw']:.4f}", color='#c0392b', fontsize=12, fontweight='bold', pad=8)
        ax0.axis('off')
        rect0 = patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--')
        ax0.add_patch(rect0)

        # 2. LLFormer (38 epochs)
        ax1 = axes[i, 1]
        ax1.imshow(res['img_pred'])
        ax1.set_title(f"LLFormer (Epoch 38 / 21.2 dB)\nPSNR: {res['psnr']:.2f} dB (▲+{res['psnr']-res['psnr_raw']:.1f} dB) | SSIM: {res['ssim']:.4f}", color='#1565c0', fontsize=12, fontweight='bold', pad=8)
        ax1.axis('off')
        rect1 = patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--')
        ax1.add_patch(rect1)

        # 3. Ground Truth
        ax2 = axes[i, 2]
        ax2.imshow(res['img_gt'])
        ax2.set_title(f"Ground Truth (Ảnh chuẩn đủ sáng)\nTarget Reference", color='#1b5e20', fontsize=12, fontweight='bold', pad=8)
        ax2.axis('off')
        rect2 = patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--')
        ax2.add_patch(rect2)

    plt.suptitle('So sánh kết quả phục hồi: Ảnh tối gốc vs LLFormer (38 Epochs) vs Ground Truth', color='#1a1a1a', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_vis_artifact = os.path.join(ARTIFACTS_DIR, "final_visual_comparison_gallery.png")
    out_vis_local = os.path.join(RESULTS_DIR, "final_visual_comparison_gallery.png")
    plt.savefig(out_vis_local, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.savefig(out_vis_artifact, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"--> Đã lưu ảnh so sánh chi tiết: {out_vis_artifact}")

    # Tạo 1 ảnh chi tiết đặc biệt (ROI Zoom & Error Map cho mẫu tiêu biểu 22.png)
    sample_hero = next((r for r in eval_results if r['fname'] == '22.png'), eval_results[0])
    plot_hero_comparison(sample_hero)

    return mean_psnr, mean_ssim, eval_results

def plot_hero_comparison(res):
    h, w = res['img_low'].shape[:2]
    bs = int(min(h, w) * 0.32)
    by, bx = int(h * 0.35), int(w * 0.35)

    crop_low = res['img_low'][by:by+bs, bx:bx+bs]
    crop_pred = res['img_pred'][by:by+bs, bx:bx+bs]
    crop_gt = res['img_gt'][by:by+bs, bx:bx+bs]
    error_map = np.abs(res['img_gt'].astype(float) - res['img_pred'].astype(float)).mean(axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor='#ffffff')
    for ax in axes.flat:
        ax.set_facecolor('#f7f9fc')
        ax.axis('off')

    # Hàng 1: Toàn cảnh
    axes[0, 0].imshow(res['img_low'])
    axes[0, 0].set_title(f"Ảnh tối gốc ({res['fname']})\nPSNR: {res['psnr_raw']:.2f} dB | SSIM: {res['ssim_raw']:.4f}", color='#c0392b', fontsize=12, fontweight='bold')
    axes[0, 0].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))

    axes[0, 1].imshow(res['img_pred'])
    axes[0, 1].set_title(f"LLFormer (Epoch 38)\nPSNR: {res['psnr']:.2f} dB | SSIM: {res['ssim']:.4f}", color='#1565c0', fontsize=12, fontweight='bold')
    axes[0, 1].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))

    axes[0, 2].imshow(res['img_gt'])
    axes[0, 2].set_title("Ground Truth (Ảnh chuẩn)", color='#1b5e20', fontsize=12, fontweight='bold')
    axes[0, 2].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))

    # Hàng 2: Zoom cận cảnh & Bản đồ sai lệch (Error Map)
    axes[1, 0].imshow(crop_low)
    axes[1, 0].set_title("Cận cảnh vùng tối gốc (Tối om, nhiễu nặng)", color='#c0392b', fontsize=11, fontweight='bold')

    axes[1, 1].imshow(crop_pred)
    axes[1, 1].set_title("Cận cảnh LLFormer phục hồi (Sáng rõ, sắc nét)", color='#1565c0', fontsize=11, fontweight='bold')

    im_err = axes[1, 2].imshow(error_map, cmap='hot')
    axes[1, 2].set_title("Bản đồ sai số |GT - LLFormer|\n(Màu tối = Khớp hoàn hảo)", color='#7b3f00', fontsize=11, fontweight='bold')
    plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Phân tích chi tiết độ khôi phục của LLFormer trên mẫu {res["fname"]}', color='#1a1a1a', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    hero_artifact = os.path.join(ARTIFACTS_DIR, "final_hero_detail_comparison.png")
    hero_local = os.path.join(RESULTS_DIR, "final_hero_detail_comparison.png")
    plt.savefig(hero_local, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.savefig(hero_artifact, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"--> Đã lưu phân tích cận cảnh: {hero_artifact}")

if __name__ == '__main__':
    plot_training_curves()
    evaluate_and_visualize()
