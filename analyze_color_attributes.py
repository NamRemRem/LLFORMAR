"""
Detailed Analysis of Hue, Saturation, and Brightness/Value
==========================================================
Investigates 3 key scientific questions:
1. Hue: Does predicted color actually shift away from Ground Truth hue?
2. Saturation: Does neon pink fade due to low saturation, or hue shift to yellow?
3. Brightness/Value: Does boosting brightness simultaneously distort color?
"""
import os
import sys
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import skimage.color as color

from model.LLFormer import LLFormer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EVAL_DIR = r"c:\Users\Admin\Documents\LLFormer\LOLdataset\eval15"
FIGURES_DIR = r"c:\Users\Admin\Documents\LLFormer\figures"
ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"

CKPT_A = r"c:\Users\Admin\Documents\LLFormer\training_50epochs_results\checkpoints\model_bestPSNR.pth"
CKPT_B = r"c:\Users\Admin\Documents\LLFormer\ablation_results\config_B_hsv\checkpoints\model_bestPSNR.pth"

def cyclic_hue_diff_deg(h1, h2):
    # h in [0, 1]
    diff = np.abs(h1 - h2)
    diff = np.minimum(diff, 1.0 - diff)
    return diff * 360.0

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
    return pred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on {device}...")
    model_a = load_model(CKPT_A, device)
    model_b = load_model(CKPT_B, device)

    # Focus on 179.png which contains the dominant neon pink region (34k+ pixels)
    img_name = "179.png"
    low_p = os.path.join(EVAL_DIR, 'low', img_name)
    high_p = os.path.join(EVAL_DIR, 'high', img_name)

    arr_low = np.array(Image.open(low_p).convert('RGB')).astype(np.float32) / 255.0
    arr_gt  = np.array(Image.open(high_p).convert('RGB')).astype(np.float32) / 255.0

    arr_a = predict(model_a, Image.open(low_p).convert('RGB'), device)
    arr_b = predict(model_b, Image.open(low_p).convert('RGB'), device)

    hsv_low = color.rgb2hsv(arr_low)
    hsv_gt  = color.rgb2hsv(arr_gt)
    hsv_a   = color.rgb2hsv(arr_a)
    hsv_b   = color.rgb2hsv(arr_b)

    # Isolate neon pink / magenta mask in GT
    # Hue: [0.78, 0.98] (approx 280° - 355°), Saturation > 0.35, Value > 0.2
    pink_mask = (
        (hsv_gt[:, :, 0] >= 0.78) & (hsv_gt[:, :, 0] <= 0.98) &
        (hsv_gt[:, :, 1] >= 0.35) &
        (hsv_gt[:, :, 2] >= 0.20)
    )
    n_pink = np.sum(pink_mask)
    print(f"Neon pink mask in {img_name}: {n_pink} pixels")

    # 1. HUE ANALYSIS
    h_gt_pink = hsv_gt[:, :, 0][pink_mask] * 360.0
    h_a_pink  = hsv_a[:, :, 0][pink_mask] * 360.0
    h_b_pink  = hsv_b[:, :, 0][pink_mask] * 360.0

    hue_err_a = cyclic_hue_diff_deg(hsv_a[:, :, 0][pink_mask], hsv_gt[:, :, 0][pink_mask])
    hue_err_b = cyclic_hue_diff_deg(hsv_b[:, :, 0][pink_mask], hsv_gt[:, :, 0][pink_mask])

    # 2. SATURATION ANALYSIS
    s_gt_pink = hsv_gt[:, :, 1][pink_mask]
    s_a_pink  = hsv_a[:, :, 1][pink_mask]
    s_b_pink  = hsv_b[:, :, 1][pink_mask]

    # 3. VALUE (BRIGHTNESS) ANALYSIS
    v_gt_pink = hsv_gt[:, :, 2][pink_mask]
    v_a_pink  = hsv_a[:, :, 2][pink_mask]
    v_b_pink  = hsv_b[:, :, 2][pink_mask]

    # Check yellow shift: yellow hue is roughly 40° - 65° (0.11 - 0.18)
    yellow_mask_a = (h_a_pink >= 35.0) & (h_a_pink <= 75.0)
    yellow_mask_b = (h_b_pink >= 35.0) & (h_b_pink <= 75.0)
    yellow_pct_a = np.mean(yellow_mask_a) * 100.0
    yellow_pct_b = np.mean(yellow_mask_b) * 100.0

    # Overall dataset statistics across all 15 images
    print("\n" + "="*70)
    print("   QUANTITATIVE BREAKDOWN FOR NEON PINK REGION (179.png)")
    print("="*70)
    print(f"Ground Truth : Mean Hue = {np.mean(h_gt_pink):.1f}° | Mean Sat = {np.mean(s_gt_pink):.3f} | Mean Val = {np.mean(v_gt_pink):.3f}")
    print(f"Baseline A   : Mean Hue = {np.mean(h_a_pink):.1f}° | Mean Sat = {np.mean(s_a_pink):.3f} | Mean Val = {np.mean(v_a_pink):.3f}")
    print(f"Config B(HSV): Mean Hue = {np.mean(h_b_pink):.1f}° | Mean Sat = {np.mean(s_b_pink):.3f} | Mean Val = {np.mean(v_b_pink):.3f}")
    print("-" * 70)
    print(f"Average Hue Deviation from GT:")
    print(f"  Baseline A  : {np.mean(hue_err_a):.1f}° (Max: {np.max(hue_err_a):.1f}°)")
    print(f"  Config B    : {np.mean(hue_err_b):.1f}° (Reduced by {np.mean(hue_err_a) - np.mean(hue_err_b):.1f}°)")
    print(f"Saturation Loss (Drop from GT {np.mean(s_gt_pink):.3f}):")
    print(f"  Baseline A  : drop of {np.mean(s_gt_pink) - np.mean(s_a_pink):.3f} ({((np.mean(s_gt_pink) - np.mean(s_a_pink))/np.mean(s_gt_pink))*100:.1f}% desaturated)")
    print(f"  Config B    : drop of {np.mean(s_gt_pink) - np.mean(s_b_pink):.3f} ({((np.mean(s_gt_pink) - np.mean(s_b_pink))/np.mean(s_gt_pink))*100:.1f}% desaturated)")
    print(f"Pixels shifted into Yellow/Orange band (35°-75°):")
    print(f"  Baseline A  : {yellow_pct_a:.1f}%")
    print(f"  Config B    : {yellow_pct_b:.1f}%")
    print("=" * 70)

    # Also compute full eval15 average HSV stats
    files = sorted([f for f in os.listdir(os.path.join(EVAL_DIR, 'low')) if f.endswith(('.png','.jpg'))])
    all_hue_err_a, all_hue_err_b = [], []
    all_sat_err_a, all_sat_err_b = [], []
    all_val_err_a, all_val_err_b = [], []

    for f in files:
        i_l = Image.open(os.path.join(EVAL_DIR, 'low', f)).convert('RGB')
        i_h = Image.open(os.path.join(EVAL_DIR, 'high', f)).convert('RGB')
        p_a = predict(model_a, i_l, device)
        p_b = predict(model_b, i_l, device)
        g   = np.array(i_h).astype(np.float32) / 255.0

        h_g = color.rgb2hsv(g); h_1 = color.rgb2hsv(p_a); h_2 = color.rgb2hsv(p_b)
        # mask out near-black or achromatic pixels where hue is undefined/noise
        valid = (h_g[:,:,2] > 0.05) & (h_g[:,:,1] > 0.1)

        all_hue_err_a.append(np.mean(cyclic_hue_diff_deg(h_1[:,:,0][valid], h_g[:,:,0][valid])))
        all_hue_err_b.append(np.mean(cyclic_hue_diff_deg(h_2[:,:,0][valid], h_g[:,:,0][valid])))

        all_sat_err_a.append(np.mean(np.abs(h_1[:,:,1][valid] - h_g[:,:,1][valid])))
        all_sat_err_b.append(np.mean(np.abs(h_2[:,:,1][valid] - h_g[:,:,1][valid])))

        all_val_err_a.append(np.mean(np.abs(h_1[:,:,2] - h_g[:,:,2])))
        all_val_err_b.append(np.mean(np.abs(h_2[:,:,2] - h_g[:,:,2])))

    print(f"\nFull Dataset (eval15) Mean Hue Error : Baseline A = {np.mean(all_hue_err_a):.2f}° | Config B = {np.mean(all_hue_err_b):.2f}°")
    print(f"Full Dataset (eval15) Mean Sat Error : Baseline A = {np.mean(all_sat_err_a):.4f} | Config B = {np.mean(all_sat_err_b):.4f}")
    print(f"Full Dataset (eval15) Mean Val Error : Baseline A = {np.mean(all_val_err_a):.4f} | Config B = {np.mean(all_val_err_b):.4f}")

    # ── GENERATE SCIENTIFIC FIGURE ──
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0d1117')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.3, wspace=0.25)

    # 1. Visual crop of 179.png neon pink
    # Bounding box around the neon sign
    y1, y2, x1, x2 = 100, 300, 150, 450
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.axis('off')
    # Combine subplots inside top row
    sub_gs = gs[0, :].subgridspec(1, 4, wspace=0.05)
    imgs = [arr_low[y1:y2, x1:x2], arr_a[y1:y2, x1:x2], arr_b[y1:y2, x1:x2], arr_gt[y1:y2, x1:x2]]
    sub_titles = [
        "Low-Light Input",
        f"Baseline A (SmoothL1)\nHue Err: {np.mean(hue_err_a):.1f}° | Sat: {np.mean(s_a_pink):.2f}",
        f"Config B (+HSV Loss)\nHue Err: {np.mean(hue_err_b):.1f}° | Sat: {np.mean(s_b_pink):.2f}",
        f"Ground Truth (Reference)\nHue: {np.mean(h_gt_pink):.1f}° | Sat: {np.mean(s_gt_pink):.2f}"
    ]
    borders = ['#ffcc00', '#ff4d4d', '#00e676', '#00b0ff']
    for idx in range(4):
        ax = fig.add_subplot(sub_gs[0, idx])
        ax.imshow(imgs[idx])
        ax.set_title(sub_titles[idx], color='white', fontsize=11, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(borders[idx]); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    # 2. Hue Distribution Plot (Bottom-Left)
    ax_hue = fig.add_subplot(gs[1, 0])
    ax_hue.set_facecolor('#161b22')
    ax_hue.hist(h_gt_pink, bins=40, range=(250, 360), alpha=0.6, color='#00b0ff', label='Ground Truth', density=True)
    ax_hue.hist(h_a_pink,  bins=40, range=(250, 360), alpha=0.5, color='#ff4d4d', label='Baseline A', density=True)
    ax_hue.hist(h_b_pink,  bins=40, range=(250, 360), alpha=0.6, color='#00e676', label='Config B (+HSV)', density=True)
    ax_hue.set_title("1. Hue Distribution (Neon Region)\n[Magenta / Pink Band: 280°-340°]", color='white', fontsize=12, fontweight='bold')
    ax_hue.set_xlabel("Hue Angle (Degrees °)", color='white', fontsize=10)
    ax_hue.set_ylabel("Probability Density", color='white', fontsize=10)
    ax_hue.tick_params(colors='white')
    ax_hue.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax_hue.grid(True, linestyle=':', alpha=0.3, color='gray')

    # 3. Saturation vs Value Scatter Plot (Bottom-Middle)
    ax_sat = fig.add_subplot(gs[1, 1])
    ax_sat.set_facecolor('#161b22')
    # Subsample for clear scatter
    sub_idx = np.random.choice(len(s_gt_pink), size=min(1500, len(s_gt_pink)), replace=False)
    ax_sat.scatter(s_a_pink[sub_idx], v_a_pink[sub_idx], s=12, alpha=0.4, color='#ff4d4d', label='Baseline A')
    ax_sat.scatter(s_b_pink[sub_idx], v_b_pink[sub_idx], s=12, alpha=0.5, color='#00e676', label='Config B (+HSV)')
    ax_sat.scatter(s_gt_pink[sub_idx], v_gt_pink[sub_idx], s=12, alpha=0.6, color='#00b0ff', label='Ground Truth')
    ax_sat.set_title("2. Saturation vs. Brightness (Value)\n[Are colors washed out by desaturation?]", color='white', fontsize=12, fontweight='bold')
    ax_sat.set_xlabel("Saturation (0 to 1)", color='white', fontsize=10)
    ax_sat.set_ylabel("Value / Brightness (0 to 1)", color='white', fontsize=10)
    ax_sat.tick_params(colors='white')
    ax_sat.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax_sat.grid(True, linestyle=':', alpha=0.3, color='gray')

    # 4. Brightness vs Hue Error Relationship (Bottom-Right)
    ax_val = fig.add_subplot(gs[1, 2])
    ax_val.set_facecolor('#161b22')
    ax_val.scatter(v_a_pink[sub_idx], hue_err_a[sub_idx], s=12, alpha=0.4, color='#ff4d4d', label='Baseline A')
    ax_val.scatter(v_b_pink[sub_idx], hue_err_b[sub_idx], s=12, alpha=0.5, color='#00e676', label='Config B (+HSV)')
    ax_val.set_title("3. Brightness vs. Hue Error\n[Does boosting Value distort Hue?]", color='white', fontsize=12, fontweight='bold')
    ax_val.set_xlabel("Predicted Brightness (Value)", color='white', fontsize=10)
    ax_val.set_ylabel("Hue Deviation from GT (Degrees °)", color='white', fontsize=10)
    ax_val.tick_params(colors='white')
    ax_val.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax_val.grid(True, linestyle=':', alpha=0.3, color='gray')

    plt.suptitle("Three-Factor Color Analysis: Hue Shift, Saturation Loss & Brightness Coupling\n"
                 "Evaluation on Neon Pink Region (Sample 179) and LOL eval15 Dataset",
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    out_fname = "color_three_factors_analysis.png"
    out_fig = os.path.join(FIGURES_DIR, out_fname)
    out_art = os.path.join(ARTIFACTS_DIR, out_fname)
    plt.savefig(out_fig, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    shutil.copy2(out_fig, out_art)
    plt.close()
    print(f"\n--> Figure saved to {out_fig} and artifacts.")

if __name__ == '__main__':
    main()
