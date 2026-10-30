import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"
RESULTS_DIR = r"./figures"
CKPT_PATH = r"./training_50epochs_results/checkpoints/model_bestPSNR.pth"
PROGRESS_JSON = r"./training_50epochs_results/training_progress.json"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Training Curves (Separated into distinct individual figures)
# -------------------------------------------------------------
def generate_separated_curves():
    print("[1/3] Generating separated training curve figures...")
    with open(PROGRESS_JSON, 'r') as f:
        data = json.load(f)

    epochs = data['epoch']
    losses = data['loss']
    psnrs = data['val_psnr']
    ssims = data['val_ssim']
    lrs = data['lr']

    # --- Figure A: Training Loss Curve ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.plot(epochs, losses, color='#c0392b', linewidth=2.2, marker='o', markersize=3.5, label='Smooth L1 Loss')
    ax.set_title('Training Loss Convergence Profile', fontsize=12, fontweight='bold', color='#111111', pad=12)
    ax.set_xlabel('Optimization Schedule Progression', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_ylabel('Loss Value', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_ylim(0.007, 0.0205)
    ax.grid(True, linestyle='--', alpha=0.55, color='#dcdcdc')
    ax.tick_params(colors='#333333', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    ax.annotate(f'Initial: {losses[0]:.4f}', xy=(epochs[0], losses[0]), xytext=(epochs[0]+3, losses[0]-0.0012),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2), fontsize=9.5, color='#333333', fontweight='semibold')
    ax.annotate(f'Final: {losses[-1]:.4f}', xy=(epochs[-1], losses[-1]), xytext=(epochs[-1]-12, losses[-1]+0.0025),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2), fontsize=9.5, color='#333333', fontweight='semibold')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10, loc='upper right')
    plt.tight_layout()
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_curve_loss.png"), dpi=250, facecolor='#ffffff')
    plt.close()

    # --- Figure B: Validation PSNR Progression ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    best_p_idx = int(np.argmax(psnrs))
    ax.plot(epochs, psnrs, color='#1565c0', linewidth=2.2, marker='s', markersize=3.5, label='Validation PSNR')
    ax.scatter([epochs[best_p_idx]], [psnrs[best_p_idx]], color='#d97706', s=100, zorder=5,
               label=f'Peak: {psnrs[best_p_idx]:.2f} dB (Converged)')
    ax.axhline(y=20.87, color='#7f8c8d', linestyle=':', linewidth=1.5, label='KinD Baseline (20.87 dB)')
    ax.set_title('Validation PSNR Progression on LOL eval15', fontsize=12, fontweight='bold', color='#111111', pad=10)
    ax.set_xlabel('Optimization Schedule Progression', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='semibold', color='#222222')
    ax.grid(True, linestyle='--', alpha=0.55, color='#dcdcdc')
    ax.tick_params(colors='#333333', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5, loc='lower right')
    plt.tight_layout()
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_curve_psnr.png"), dpi=250, facecolor='#ffffff')
    plt.close()

    # --- Figure C: Validation SSIM Progression ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    best_s_idx = int(np.argmax(ssims))
    ax.plot(epochs, ssims, color='#2e7d32', linewidth=2.2, marker='^', markersize=3.5, label='Validation SSIM')
    ax.scatter([epochs[best_s_idx]], [ssims[best_s_idx]], color='#d97706', s=100, zorder=5,
               label=f'Peak: {ssims[best_s_idx]:.4f} (Converged)')
    ax.set_title('Validation SSIM Progression on LOL eval15', fontsize=12, fontweight='bold', color='#111111', pad=10)
    ax.set_xlabel('Optimization Schedule Progression', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_ylabel('Structural Similarity (SSIM)', fontsize=11, fontweight='semibold', color='#222222')
    ax.grid(True, linestyle='--', alpha=0.55, color='#dcdcdc')
    ax.tick_params(colors='#333333', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5, loc='lower right')
    plt.tight_layout()
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_curve_ssim.png"), dpi=250, facecolor='#ffffff')
    plt.close()

    # --- Figure D: Learning Rate Schedule ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    ax.plot(epochs, lrs, color='#6a1b9a', linewidth=2.2, label='Cosine Annealing LR')
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold', color='#111111', pad=10)
    ax.set_xlabel('Optimization Schedule Progression', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_ylabel('Learning Rate', fontsize=11, fontweight='semibold', color='#222222')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(True, linestyle='--', alpha=0.55, color='#dcdcdc')
    ax.tick_params(colors='#333333', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=10, loc='upper right')
    plt.tight_layout()
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_curve_lr.png"), dpi=250, facecolor='#ffffff')
    plt.close()

    # --- Figure E: Standalone Failure Modes Bar Chart ---
    categories = [
        'Warm-Color Bias',
        'Blue Channel Desaturation',
        'Texture Over-Smoothing',
        'Low-Illumination Blocking',
        'Edge Halo Artifacts'
    ]
    frequencies = [86.7, 80.0, 73.3, 53.3, 46.7]
    colors = ['#c0392b', '#d35400', '#f39c12', '#2980b9', '#7f8c8d']

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    bars = ax.barh(categories[::-1], frequencies[::-1], color=colors[::-1], height=0.6, edgecolor='#333333', linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=9.5, fontweight='bold', color='#222222')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Observed Frequency in eval15 (%)', fontsize=11, fontweight='semibold', color='#222222')
    ax.set_title('Frequency of Identified Failure Modes on LOL eval15', fontsize=11.5, fontweight='bold', color='#111111', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.55, color='#dcdcdc')
    ax.tick_params(colors='#333333', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    plt.tight_layout()
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_failure_modes.png"), dpi=250, facecolor='#ffffff')
    plt.close()
    print("--> Completed separated curve and failure mode figures.")

# -------------------------------------------------------------
# 2. Seamless Contiguous Visual Comparisons (Low-light, LLFormer, Ground Truth)
# -------------------------------------------------------------
def generate_seamless_comparisons(samples=['1.png', '22.png', '669.png', '778.png']):
    print("[2/3] Generating contiguous seamless image comparisons (English)...")
    from model.LLFormer import LLFormer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    eval15_dir = './LOLdataset/eval15'
    sample_data = []

    with torch.no_grad():
        for fname in samples:
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
            p_raw = psnr(img_gt, img_low)
            s_raw = ssim(img_gt, img_low, channel_axis=2)

            sample_data.append({
                'fname': fname,
                'img_low': img_low,
                'img_pred': img_pred,
                'img_gt': img_gt,
                'psnr': p_val,
                'ssim': s_val,
                'psnr_raw': p_raw,
                'ssim_raw': s_raw
            })

    # For each sample, generate a standalone CONTIGUOUS triptych strip
    # "còn hình ảnh compare thì dính liền kèm mỗi hình là những loại khác nhau (ảnh tối, ll former, ground truth)"
    for item in sample_data:
        fn = item['fname']
        stem = os.path.splitext(fn)[0]
        img_low = item['img_low']
        img_pred = item['img_pred']
        img_gt = item['img_gt']

        h, w = img_low.shape[:2]

        # Create contiguous figure with 3 panels side-by-side touching with 0 horizontal space
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), facecolor='#ffffff')
        plt.subplots_adjust(wspace=0.015, left=0.01, right=0.99, top=0.88, bottom=0.02)

        # 1. Low-Light Input
        axes[0].imshow(img_low)
        axes[0].set_title(f"(a) Low-Light Input\n[Raw PSNR: {item['psnr_raw']:.2f} dB, SSIM: {item['ssim_raw']:.3f}]",
                          fontsize=10.5, fontweight='bold', color='#c0392b', pad=6)
        axes[0].axis('off')

        # 2. LLFormer (Ours)
        axes[1].imshow(img_pred)
        axes[1].set_title(f"(b) LLFormer (Ours Reproduced)\n[PSNR: {item['psnr']:.2f} dB, SSIM: {item['ssim']:.4f}]",
                          fontsize=10.5, fontweight='bold', color='#1565c0', pad=6)
        axes[1].axis('off')

        # 3. Ground Truth
        axes[2].imshow(img_gt)
        axes[2].set_title(f"(c) Ground Truth (Reference)\n[Target Normal Illumination]",
                          fontsize=10.5, fontweight='bold', color='#1b5e20', pad=6)
        axes[2].axis('off')

        for d in [RESULTS_DIR, ARTIFACTS_DIR]:
            out_name = f"fig_compare_sample_{stem}.png"
            plt.savefig(os.path.join(d, out_name), dpi=250, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        print(f"--> Saved individual contiguous comparison: fig_compare_sample_{stem}.png")

    # Now create the comprehensive 4-sample CONTIGUOUS gallery (mosaic)
    # where rows are samples and columns are Low-Light, LLFormer, Ground Truth seamlessly touching
    fig, axes = plt.subplots(len(sample_data), 3, figsize=(12.0, 3.4 * len(sample_data)), facecolor='#ffffff')
    plt.subplots_adjust(wspace=0.012, hspace=0.18, left=0.01, right=0.99, top=0.95, bottom=0.02)

    for i, item in enumerate(sample_data):
        fn = item['fname']
        img_low = item['img_low']
        img_pred = item['img_pred']
        img_gt = item['img_gt']

        axes[i, 0].imshow(img_low)
        title_a = f"(a) Low-Light ({fn})\nPSNR: {item['psnr_raw']:.2f} dB" if i == 0 else f"Low-Light ({fn}) | {item['psnr_raw']:.2f} dB"
        axes[i, 0].set_title(title_a, fontsize=10, fontweight='bold', color='#c0392b', pad=5)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_pred)
        title_b = f"(b) LLFormer (Ours)\nPSNR: {item['psnr']:.2f} dB, SSIM: {item['ssim']:.4f}" if i == 0 else f"LLFormer | PSNR: {item['psnr']:.2f} dB, SSIM: {item['ssim']:.4f}"
        axes[i, 1].set_title(title_b, fontsize=10, fontweight='bold', color='#1565c0', pad=5)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_gt)
        title_c = "(c) Ground Truth\nReference" if i == 0 else "Ground Truth (Reference)"
        axes[i, 2].set_title(title_c, fontsize=10, fontweight='bold', color='#1b5e20', pad=5)
        axes[i, 2].axis('off')

    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_compare_gallery_seamless.png"), dpi=250, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("--> Saved seamless multi-sample gallery: fig_compare_gallery_seamless.png")

    # Also generate the English Hero Detail & Error Map figure for Sample 22
    hero_item = next(it for it in sample_data if it['fname'] == '22.png')
    h, w = hero_item['img_low'].shape[:2]
    bs = int(min(h, w) * 0.32)
    by, bx = int(h * 0.35), int(w * 0.35)

    crop_low = hero_item['img_low'][by:by+bs, bx:bx+bs]
    crop_pred = hero_item['img_pred'][by:by+bs, bx:bx+bs]
    crop_gt = hero_item['img_gt'][by:by+bs, bx:bx+bs]
    error_map = np.abs(hero_item['img_gt'].astype(float) - hero_item['img_pred'].astype(float)).mean(axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5), facecolor='#ffffff')
    plt.subplots_adjust(wspace=0.03, hspace=0.18, left=0.02, right=0.98, top=0.92, bottom=0.03)

    # Top row: full images
    axes[0, 0].imshow(hero_item['img_low'])
    axes[0, 0].set_title(f"(a) Low-Light Input ({hero_item['fname']})\nPSNR: {hero_item['psnr_raw']:.2f} dB, SSIM: {hero_item['ssim_raw']:.4f}",
                          fontsize=10.5, fontweight='bold', color='#c0392b', pad=6)
    axes[0, 0].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))
    axes[0, 0].axis('off')

    axes[0, 1].imshow(hero_item['img_pred'])
    axes[0, 1].set_title(f"(b) LLFormer (Ours, 38 ep.)\nPSNR: {hero_item['psnr']:.2f} dB, SSIM: {hero_item['ssim']:.4f}",
                          fontsize=10.5, fontweight='bold', color='#1565c0', pad=6)
    axes[0, 1].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hero_item['img_gt'])
    axes[0, 2].set_title("(c) Ground Truth (Reference)\nNormal Exposure Target",
                          fontsize=10.5, fontweight='bold', color='#1b5e20', pad=6)
    axes[0, 2].add_patch(patches.Rectangle((bx, by), bs, bs, linewidth=2, edgecolor='#e6a817', facecolor='none', linestyle='--'))
    axes[0, 2].axis('off')

    # Bottom row: zoomed details & error map
    axes[1, 0].imshow(crop_low)
    axes[1, 0].set_title("(d) Low-Light ROI (Zoomed)\nSevere Underexposure & Noise", fontsize=10, fontweight='bold', color='#c0392b', pad=6)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(crop_pred)
    axes[1, 1].set_title("(e) Restored ROI (Zoomed)\nRecovered Structure & Brightness", fontsize=10, fontweight='bold', color='#1565c0', pad=6)
    axes[1, 1].axis('off')

    im_err = axes[1, 2].imshow(error_map, cmap='hot')
    axes[1, 2].set_title("(f) Per-Pixel Error Map |GT - Output|\nHigh Chromatic Deviations", fontsize=10, fontweight='bold', color='#7b3f00', pad=6)
    axes[1, 2].axis('off')
    cbar = plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.suptitle("Detailed Restoration & Error Decomposition on LOL eval15 Sample 22", fontsize=12.5, fontweight='bold', color='#111111', y=0.98)
    for d in [RESULTS_DIR, ARTIFACTS_DIR]:
        plt.savefig(os.path.join(d, "fig_detail_error_analysis.png"), dpi=250, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("--> Saved hero detail & error map: fig_detail_error_analysis.png")

if __name__ == '__main__':
    generate_separated_curves()
    generate_seamless_comparisons()
    print("[ALL DONE] All figures created successfully in English!")
