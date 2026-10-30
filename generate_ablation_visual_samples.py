"""
Ablation Visual Samples Generator
=================================
Generate side-by-side visual comparisons:
Low-light Input | Baseline A | Config B (+HSV Loss) | Ground Truth
With metrics (PSNR, SSIM, DeltaE00) and zoomed crop insets.
"""
import os
import sys
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
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import skimage.color as color

from model.LLFormer import LLFormer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EVAL_DIR = r"c:\Users\Admin\Documents\LLFormer\LOLdataset\eval15"
FIGURES_DIR = r"c:\Users\Admin\Documents\LLFormer\figures"
ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"

CKPT_A = r"c:\Users\Admin\Documents\LLFormer\training_50epochs_results\checkpoints\model_bestPSNR.pth"
CKPT_B = r"c:\Users\Admin\Documents\LLFormer\ablation_results\config_B_hsv\checkpoints\model_bestPSNR.pth"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def calc_delta_e00(pred_rgb, gt_rgb):
    pred_lab = color.rgb2lab(pred_rgb)
    gt_lab = color.rgb2lab(gt_rgb)
    de00 = color.deltaE_ciede2000(pred_lab, gt_lab)
    return float(np.mean(de00)), de00

def load_model(weights_path, device):
    m = LLFormer(
        inp_channels=3, out_channels=3, dim=16,
        num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', attention=True, skip=False
    ).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m

def predict(model, img_pil, device):
    t_l = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    h, w = t_l.shape[2], t_l.shape[3]
    mul = 16
    H, W = ((h + mul - 1) // mul) * mul, ((w + mul - 1) // mul) * mul
    pad = F.pad(t_l, (0, W - w, 0, H - h), 'reflect')
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred = model(pad)
    pred = torch.clamp(pred[:, :, :h, :w], 0, 1)
    arr = pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return arr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Using device: {device}")
    
    print("--> Loading Baseline Model A...")
    model_a = load_model(CKPT_A, device)
    print("--> Loading Config B (+HSV Loss) Model...")
    model_b = load_model(CKPT_B, device)

    # Key samples to analyze
    # (image_name, crop_box [ymin, xmin, ymax, xmax] relative 0-1)
    samples = [
        ("111.png", (0.35, 0.40, 0.65, 0.70), "Sample 111 (Interior Room - Color Fidelity)"),
        ("179.png", (0.25, 0.35, 0.55, 0.65), "Sample 179 (Extremely Low Light - Hue Recovery)"),
        ("23.png",  (0.40, 0.20, 0.70, 0.50), "Sample 23 (Bookshelf & Objects - High Saturation)"),
        ("669.png", (0.30, 0.45, 0.60, 0.75), "Sample 669 (Hallway & Walls - Texture & Tone)")
    ]

    gallery_data = []

    for img_name, crop_rel, title in samples:
        print(f"\n--> Processing {img_name}...")
        low_p = os.path.join(EVAL_DIR, 'low', img_name)
        high_p = os.path.join(EVAL_DIR, 'high', img_name)
        img_low = Image.open(low_p).convert('RGB')
        img_high = Image.open(high_p).convert('RGB')

        arr_gt = np.array(img_high).astype(np.float32) / 255.0
        arr_low = np.array(img_low).astype(np.float32) / 255.0

        arr_a = predict(model_a, img_low, device)
        arr_b = predict(model_b, img_low, device)

        # Metrics for A
        psnr_a = psnr_fn(arr_gt, arr_a, data_range=1.0)
        ssim_a = ssim_fn(arr_gt, arr_a, data_range=1.0, channel_axis=2)
        de00_a, err_map_a = calc_delta_e00(arr_a, arr_gt)

        # Metrics for B
        psnr_b = psnr_fn(arr_gt, arr_b, data_range=1.0)
        ssim_b = ssim_fn(arr_gt, arr_b, data_range=1.0, channel_axis=2)
        de00_b, err_map_b = calc_delta_e00(arr_b, arr_gt)

        H, W = arr_gt.shape[0], arr_gt.shape[1]
        y1, x1 = int(crop_rel[0] * H), int(crop_rel[1] * W)
        y2, x2 = int(crop_rel[2] * H), int(crop_rel[3] * W)

        crop_low = arr_low[y1:y2, x1:x2]
        crop_a   = arr_a[y1:y2, x1:x2]
        crop_b   = arr_b[y1:y2, x1:x2]
        crop_gt  = arr_gt[y1:y2, x1:x2]

        gallery_data.append({
            'name': img_name,
            'title': title,
            'low': arr_low, 'a': arr_a, 'b': arr_b, 'gt': arr_gt,
            'err_a': err_map_a, 'err_b': err_map_b,
            'm_a': (psnr_a, ssim_a, de00_a),
            'm_b': (psnr_b, ssim_b, de00_b),
            'crop_box': (x1, y1, x2 - x1, y2 - y1),
            'crops': (crop_low, crop_a, crop_b, crop_gt)
        })

        # ── Create Individual Comparison Figure (4 columns + zoom insets) ──
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 2]})
        fig.patch.set_facecolor('#0d1117')

        titles = [
            "Low-Light Input",
            f"Baseline A (SmoothL1)\nPSNR: {psnr_a:.2f} dB | SSIM: {ssim_a:.4f} | ΔE₀₀: {de00_a:.2f}",
            f"Config B (+ HSV Chroma Loss)\nPSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f} | ΔE₀₀: {de00_b:.2f}",
            "Ground Truth (Reference)"
        ]
        images = [arr_low, arr_a, arr_b, arr_gt]
        crops  = [crop_low, crop_a, crop_b, crop_gt]
        box_colors = ['#ffcc00', '#ff4d4d', '#00e676', '#00b0ff']

        for col in range(4):
            # Top row: full image with crop rectangle
            ax = axes[0, col]
            ax.imshow(images[col])
            ax.set_title(titles[col], color='white', fontsize=12, pad=10, fontweight='bold')
            ax.axis('off')
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2.5, edgecolor=box_colors[col], facecolor='none', linestyle='--')
            ax.add_patch(rect)

            # Bottom row: zoomed crop
            ax_crop = axes[1, col]
            ax_crop.imshow(crops[col])
            label = "Zoomed Region" if col == 0 else f"Zoomed Region\n{'ΔE₀₀: ' + f'{de00_a:.2f}' if col==1 else ('ΔE₀₀: ' + f'{de00_b:.2f}' if col==2 else 'Target Color')}"
            ax_crop.set_title(label, color='white', fontsize=11, pad=6)
            for spine in ax_crop.spines.values():
                spine.set_edgecolor(box_colors[col])
                spine.set_linewidth(2.5)
            ax_crop.set_xticks([])
            ax_crop.set_yticks([])

        plt.suptitle(f"Ablation Visual Comparison: {title}\nBaseline A (RGB Loss) vs. Config B (RGB + HSV Loss)",
                     color='white', fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()

        out_fname = f"ablation_visual_sample_{img_name.replace('.png','')}.png"
        out_fig = os.path.join(FIGURES_DIR, out_fname)
        out_art = os.path.join(ARTIFACTS_DIR, out_fname)
        plt.savefig(out_fig, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        shutil.copy2(out_fig, out_art)
        plt.close()
        print(f"--> Saved individual sample figure: {out_fname}")

    # ── Create Comprehensive 4-Scene Gallery Figure ──
    print("\n--> Generating 4-Scene Ablation Comparison Gallery...")
    fig, axes = plt.subplots(4, 5, figsize=(24, 18))
    fig.patch.set_facecolor('#0d1117')

    col_headers = [
        "Low-Light Input",
        "Baseline A (SmoothL1)",
        "Config B (+ HSV Loss)",
        "Ground Truth",
        "Color Error Map (ΔE₀₀ Diff)"
    ]

    for row, data in enumerate(gallery_data):
        # 0: Low-light
        axes[row, 0].imshow(data['low'])
        axes[row, 0].axis('off')

        # 1: Baseline A
        p_a, s_a, de_a = data['m_a']
        axes[row, 1].imshow(data['a'])
        axes[row, 1].axis('off')
        axes[row, 1].text(0.02, 0.05, f"PSNR: {p_a:.2f}dB | ΔE₀₀: {de_a:.2f}",
                          transform=axes[row, 1].transAxes, color='white',
                          bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7), fontsize=9)

        # 2: Config B
        p_b, s_b, de_b = data['m_b']
        diff_de = de_b - de_a
        diff_str = f"({diff_de:+.2f} ΔE)"
        axes[row, 2].imshow(data['b'])
        axes[row, 2].axis('off')
        axes[row, 2].text(0.02, 0.05, f"PSNR: {p_b:.2f}dB | ΔE₀₀: {de_b:.2f} {diff_str}",
                          transform=axes[row, 2].transAxes, color='#00e676' if diff_de < 0 else 'white',
                          bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7), fontsize=9, fontweight='bold')

        # 3: Ground Truth
        axes[row, 3].imshow(data['gt'])
        axes[row, 3].axis('off')

        # 4: Error difference map: where Config B has lower error than Baseline A (green) vs higher (magenta)
        # err_map: de00 per pixel. Delta = de00_b - de00_a (negative is good)
        err_diff = data['err_b'] - data['err_a']
        im_err = axes[row, 4].imshow(err_diff, cmap='coolwarm', vmin=-10, vmax=10)
        axes[row, 4].axis('off')
        axes[row, 4].text(0.02, 0.05, f"Blue=HSV better | Red=Base better",
                          transform=axes[row, 4].transAxes, color='white',
                          bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7), fontsize=9)

        # Row label on left
        axes[row, 0].text(-0.08, 0.5, data['name'], transform=axes[row, 0].transAxes,
                          color='white', fontsize=12, fontweight='bold', va='center', rotation=90)

    for col in range(5):
        axes[0, col].set_title(col_headers[col], color='white', fontsize=13, fontweight='bold', pad=12)

    plt.suptitle("LLFormer Ablation Study: Visual Comparison Across Representative LOL Test Scenes\n"
                 "Baseline A vs. Config B (+HSV Chroma Loss) vs. Ground Truth Reference",
                 color='white', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    gal_fname = "ablation_visual_gallery_AB.png"
    gal_fig = os.path.join(FIGURES_DIR, gal_fname)
    gal_art = os.path.join(ARTIFACTS_DIR, gal_fname)
    plt.savefig(gal_fig, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    shutil.copy2(gal_fig, gal_art)
    plt.close()
    print(f"--> Saved visual gallery figure: {gal_fname}")
    print("\n[DONE] All ablation visual sample figures created successfully!")

if __name__ == '__main__':
    main()
