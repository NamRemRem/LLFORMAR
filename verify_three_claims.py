"""
Rigorous Verification of 3 Scientific Claims Before Paper Writing
================================================================
1. Shortest circular Hue distance Delta H stability under varying saturation/value thresholds.
2. Saturation measurement strictly inside the Neon Pink ROI / Mask.
3. Verification of "overexposure" claim via clipping ratio and gradient magnitude / texture preservation.
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import skimage.color as color
from scipy.ndimage import sobel

from model.LLFormer import LLFormer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EVAL_DIR = r"c:\Users\Admin\Documents\LLFormer\LOLdataset\eval15"
FIGURES_DIR = r"c:\Users\Admin\Documents\LLFormer\figures"
ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"

CKPT_A = r"c:\Users\Admin\Documents\LLFormer\training_50epochs_results\checkpoints\model_bestPSNR.pth"
CKPT_B = r"c:\Users\Admin\Documents\LLFormer\ablation_results\config_B_hsv\checkpoints\model_bestPSNR.pth"

def shortest_circular_hue_diff_deg(h1_deg, h2_deg):
    """
    Computes shortest angular distance on the color circle [0, 360):
    Delta H = min(|h1 - h2|, 360 - |h1 - h2|)
    """
    diff = np.abs(h1_deg - h2_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)

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

def compute_gradient_magnitude(img_gray):
    sx = sobel(img_gray, axis=1, mode='reflect')
    sy = sobel(img_gray, axis=0, mode='reflect')
    return np.hypot(sx, sy)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading models...")
    model_a = load_model(CKPT_A, device)
    model_b = load_model(CKPT_B, device)

    # Load 179.png
    img_name = "179.png"
    low_p = os.path.join(EVAL_DIR, 'low', img_name)
    high_p = os.path.join(EVAL_DIR, 'high', img_name)

    img_low = Image.open(low_p).convert('RGB')
    img_gt  = Image.open(high_p).convert('RGB')

    arr_low = np.array(img_low).astype(np.float32) / 255.0
    arr_gt  = np.array(img_gt).astype(np.float32) / 255.0

    arr_a = predict(model_a, img_low, device)
    arr_b = predict(model_b, img_low, device)

    hsv_gt = color.rgb2hsv(arr_gt)
    hsv_a  = color.rgb2hsv(arr_a)
    hsv_b  = color.rgb2hsv(arr_b)

    H_gt_deg = hsv_gt[:, :, 0] * 360.0
    H_a_deg  = hsv_a[:, :, 0] * 360.0
    H_b_deg  = hsv_b[:, :, 0] * 360.0

    S_gt = hsv_gt[:, :, 1]
    S_a  = hsv_a[:, :, 1]
    S_b  = hsv_b[:, :, 1]

    V_gt = hsv_gt[:, :, 2]
    V_a  = hsv_a[:, :, 2]
    V_b  = hsv_b[:, :, 2]

    # Pixel-wise shortest angular hue diff
    delta_h_a = shortest_circular_hue_diff_deg(H_a_deg, H_gt_deg)
    delta_h_b = shortest_circular_hue_diff_deg(H_b_deg, H_gt_deg)

    # =========================================================================
    # CHECK 1: HUE ERROR UNDER VARIOUS SATURATION & VALUE THRESHOLDS
    # =========================================================================
    print("\n" + "="*85)
    print("   CHECK 1: DELTA H ROBUSTNESS ACROSS SATURATION & VALUE THRESHOLDS (179.png)")
    print("="*85)
    print(f"{'Condition / Filter':<32} | {'Valid %':>7} | {'Base A Mean':>11} | {'Base A Med':>10} | {'Cfg B Mean':>11} | {'Cfg B Med':>10} | {'Advantage'}")
    print("-" * 85)

    filters = [
        ("All pixels (No filter)", np.ones_like(S_gt, dtype=bool)),
        ("S_gt > 0.05 & V_gt > 0.05", (S_gt > 0.05) & (V_gt > 0.05)),
        ("S_gt > 0.10 & V_gt > 0.05", (S_gt > 0.10) & (V_gt > 0.05)),
        ("S_gt > 0.10 & V_gt > 0.10", (S_gt > 0.10) & (V_gt > 0.10)),
        ("S_gt > 0.20 & V_gt > 0.10", (S_gt > 0.20) & (V_gt > 0.10)),
        ("S_gt > 0.30 & V_gt > 0.10", (S_gt > 0.30) & (V_gt > 0.10)),
        ("S_gt > 0.50 & V_gt > 0.15", (S_gt > 0.50) & (V_gt > 0.15)),
    ]

    for name, mask in filters:
        pct = np.mean(mask) * 100.0
        m_a = np.mean(delta_h_a[mask])
        med_a = np.median(delta_h_a[mask])
        m_b = np.mean(delta_h_b[mask])
        med_b = np.median(delta_h_b[mask])
        adv = f"B is {- (m_b - m_a):+.1f}° better"
        print(f"{name:<32} | {pct:>6.1f}% | {m_a:>10.2f}° | {med_a:>9.2f}° | {m_b:>10.2f}° | {med_b:>9.2f}° | {adv}")

    # Also compute across all 15 eval images
    print("\n--> Evaluating Check 1 on entire eval15 dataset (S_gt > 0.1 & V_gt > 0.1):")
    eval_files = sorted([f for f in os.listdir(os.path.join(EVAL_DIR, 'low')) if f.endswith(('.png','.jpg'))])
    all_dh_a, all_dh_b = [], []
    for f in eval_files:
        i_l = Image.open(os.path.join(EVAL_DIR, 'low', f)).convert('RGB')
        i_h = Image.open(os.path.join(EVAL_DIR, 'high', f)).convert('RGB')
        p_a = predict(model_a, i_l, device)
        p_b = predict(model_b, i_l, device)
        g   = np.array(i_h).astype(np.float32) / 255.0

        h_g = color.rgb2hsv(g); h_1 = color.rgb2hsv(p_a); h_2 = color.rgb2hsv(p_b)
        vmask = (h_g[:,:,1] > 0.10) & (h_g[:,:,2] > 0.10)

        dh1 = shortest_circular_hue_diff_deg(h_1[:,:,0][vmask]*360.0, h_g[:,:,0][vmask]*360.0)
        dh2 = shortest_circular_hue_diff_deg(h_2[:,:,0][vmask]*360.0, h_g[:,:,0][vmask]*360.0)
        all_dh_a.append((np.mean(dh1), np.median(dh1)))
        all_dh_b.append((np.mean(dh2), np.median(dh2)))

    mean_dh_a = np.mean([x[0] for x in all_dh_a])
    med_dh_a  = np.median([x[1] for x in all_dh_a])
    mean_dh_b = np.mean([x[0] for x in all_dh_b])
    med_dh_b  = np.median([x[1] for x in all_dh_b])
    print(f"Entire eval15 Mean Hue Error   : Baseline A = {mean_dh_a:.2f}° | Config B = {mean_dh_b:.2f}° (Δ = {mean_dh_b - mean_dh_a:+.2f}°)")
    print(f"Entire eval15 Median Hue Error : Baseline A = {med_dh_a:.2f}° | Config B = {med_dh_b:.2f}°")

    # =========================================================================
    # CHECK 2: MEASURE SATURATION STRICTLY INSIDE THE NEON PINK ROI / MASK
    # =========================================================================
    print("\n" + "="*85)
    print("   CHECK 2: SATURATION DISTRIBUTION STRICTLY INSIDE NEON PINK MASK (179.png)")
    print("="*85)
    # Define exact Neon Pink mask on GT:
    # Hue: [280°, 355°] (magenta/pink), Saturation >= 0.35, Value >= 0.15
    neon_mask = (H_gt_deg >= 280.0) & (H_gt_deg <= 355.0) & (S_gt >= 0.35) & (V_gt >= 0.15)
    n_neon = np.sum(neon_mask)
    neon_pct = n_neon / neon_mask.size * 100.0
    print(f"Neon Pink mask size: {n_neon} pixels ({neon_pct:.2f}% of image)")

    s_gt_neon = S_gt[neon_mask]
    s_a_neon  = S_a[neon_mask]
    s_b_neon  = S_b[neon_mask]

    stats = [
        ("Ground Truth", s_gt_neon),
        ("Baseline A (SmoothL1)", s_a_neon),
        ("Config B (+ HSV Loss)", s_b_neon),
    ]
    print(f"{'Method':<25} | {'Mean':>7} | {'Median':>7} | {'Std':>7} | {'Q25':>7} | {'Q75':>7} | {'Min':>7} | {'Max':>7}")
    print("-" * 85)
    for name, vals in stats:
        print(f"{name:<25} | {np.mean(vals):>7.3f} | {np.median(vals):>7.3f} | {np.std(vals):>7.3f} | "
              f"{np.percentile(vals, 25):>7.3f} | {np.percentile(vals, 75):>7.3f} | {np.min(vals):>7.3f} | {np.max(vals):>7.3f}")

    rec_ratio_a = np.mean(s_a_neon) / np.mean(s_gt_neon) * 100.0
    rec_ratio_b = np.mean(s_b_neon) / np.mean(s_gt_neon) * 100.0
    print(f"\nSaturation Recovery Ratio (Mean Pred / Mean GT):")
    print(f"  Baseline A : {rec_ratio_a:.1f}%")
    print(f"  Config B   : {rec_ratio_b:.1f}%")

    # Check whether the hue inside this mask rotated to yellow in Baseline vs Config B
    h_a_neon = H_a_deg[neon_mask]
    h_b_neon = H_b_deg[neon_mask]
    yellow_a = np.mean((h_a_neon >= 35.0) & (h_a_neon <= 75.0)) * 100.0
    yellow_b = np.mean((h_b_neon >= 35.0) & (h_b_neon <= 75.0)) * 100.0
    pink_retained_a = np.mean((h_a_neon >= 280.0) | (h_a_neon <= 20.0)) * 100.0
    pink_retained_b = np.mean((h_b_neon >= 280.0) | (h_b_neon <= 20.0)) * 100.0
    print(f"Hue classification inside Neon Mask:")
    print(f"  Retained Pink/Red (Hue >= 280° or <= 20°): Baseline A = {pink_retained_a:.1f}% | Config B = {pink_retained_b:.1f}%")
    print(f"  Shifted to Yellow/Orange (Hue 35°-75°)   : Baseline A = {yellow_a:.1f}% | Config B = {yellow_b:.1f}%")

    # =========================================================================
    # CHECK 3: VERIFY "OVEREXPOSURE" (CLIPPING & HIGH-LIGHT TEXTURE LOSS)
    # =========================================================================
    print("\n" + "="*85)
    print("   CHECK 3: VERIFICATION OF OVEREXPOSURE & HIGH-LIGHT CLIPPING (179.png & eval15)")
    print("="*85)

    def check_clipping(arr_rgb, v_channel):
        clip_v95 = np.mean(v_channel >= 0.95) * 100.0
        clip_v98 = np.mean(v_channel >= 0.98) * 100.0
        clip_v10 = np.mean(v_channel >= 0.999) * 100.0
        any_channel_clip = np.mean(np.any(arr_rgb >= 0.98, axis=-1)) * 100.0
        all_channel_clip = np.mean(np.all(arr_rgb >= 0.98, axis=-1)) * 100.0
        return clip_v95, clip_v98, clip_v10, any_channel_clip, all_channel_clip

    # 1. On 179.png (Full Image and Neon ROI)
    print("1. Clipping Analysis on 179.png Full Image:")
    c_gt = check_clipping(arr_gt, V_gt)
    c_a  = check_clipping(arr_a,  V_a)
    c_b  = check_clipping(arr_b,  V_b)

    print(f"{'Method':<25} | {'V >= 0.95':>10} | {'V >= 0.98':>10} | {'V == 1.0 (hard)':>15} | {'Any RGB >= 0.98':>16} | {'White Blown (All>=0.98)'}")
    print("-" * 95)
    for name, c in [("Ground Truth", c_gt), ("Baseline A", c_a), ("Config B", c_b)]:
        print(f"{name:<25} | {c[0]:>9.3f}% | {c[1]:>9.3f}% | {c[2]:>14.3f}% | {c[3]:>15.3f}% | {c[4]:>10.3f}%")

    # 2. Inside Neon Highlight Mask (179.png)
    print("\n2. Clipping inside Neon Region (179.png):")
    c_gt_n = check_clipping(arr_gt[neon_mask], V_gt[neon_mask])
    c_a_n  = check_clipping(arr_a[neon_mask],  V_a[neon_mask])
    c_b_n  = check_clipping(arr_b[neon_mask],  V_b[neon_mask])
    for name, c in [("Ground Truth", c_gt_n), ("Baseline A", c_a_n), ("Config B", c_b_n)]:
        print(f"{name:<25} | V>=0.95: {c[0]:>5.2f}% | V>=0.98: {c[1]:>5.2f}% | Hard Clip: {c[2]:>5.2f}% | Any RGB Clip: {c[3]:>5.2f}%")

    # 3. High-light Texture / Gradient Magnitude Analysis
    gray_gt = color.rgb2gray(arr_gt)
    gray_a  = color.rgb2gray(arr_a)
    gray_b  = color.rgb2gray(arr_b)

    grad_gt = compute_gradient_magnitude(gray_gt)
    grad_a  = compute_gradient_magnitude(gray_a)
    grad_b  = compute_gradient_magnitude(gray_b)

    # Compare gradient magnitude in highlights (V_gt > 0.5)
    hl_mask = V_gt > 0.5
    print(f"\n3. Texture / Gradient Magnitude in Highlight Regions (V_gt > 0.5, {np.sum(hl_mask)} px):")
    print(f"  Ground Truth Gradient Mean : {np.mean(grad_gt[hl_mask]):.4f} (Std: {np.std(grad_gt[hl_mask]):.4f})")
    print(f"  Baseline A   Gradient Mean : {np.mean(grad_a[hl_mask]):.4f} (Ratio to GT: {np.mean(grad_a[hl_mask])/np.mean(grad_gt[hl_mask])*100:.1f}%)")
    print(f"  Config B     Gradient Mean : {np.mean(grad_b[hl_mask]):.4f} (Ratio to GT: {np.mean(grad_b[hl_mask])/np.mean(grad_gt[hl_mask])*100:.1f}%)")

    # In neon mask specifically
    print(f"\n4. Texture / Gradient Magnitude in Neon Mask:")
    print(f"  Ground Truth Gradient Mean : {np.mean(grad_gt[neon_mask]):.4f}")
    print(f"  Baseline A   Gradient Mean : {np.mean(grad_a[neon_mask]):.4f} (Ratio: {np.mean(grad_a[neon_mask])/np.mean(grad_gt[neon_mask])*100:.1f}%)")
    print(f"  Config B     Gradient Mean : {np.mean(grad_b[neon_mask]):.4f} (Ratio: {np.mean(grad_b[neon_mask])/np.mean(grad_gt[neon_mask])*100:.1f}%)")

    # 4. Across entire eval15 dataset clipping stats
    print("\n5. Clipping statistics across all 15 eval images (eval15 average):")
    eval_clip_gt, eval_clip_a, eval_clip_b = [], [], []
    for f in eval_files:
        i_l = Image.open(os.path.join(EVAL_DIR, 'low', f)).convert('RGB')
        i_h = Image.open(os.path.join(EVAL_DIR, 'high', f)).convert('RGB')
        p_a = predict(model_a, i_l, device)
        p_b = predict(model_b, i_l, device)
        g   = np.array(i_h).astype(np.float32) / 255.0

        v_g = color.rgb2hsv(g)[:,:,2]
        v_1 = color.rgb2hsv(p_a)[:,:,2]
        v_2 = color.rgb2hsv(p_b)[:,:,2]

        eval_clip_gt.append(check_clipping(g, v_g))
        eval_clip_a.append(check_clipping(p_a, v_1))
        eval_clip_b.append(check_clipping(p_b, v_2))

    mean_c_gt = np.mean(eval_clip_gt, axis=0)
    mean_c_a  = np.mean(eval_clip_a, axis=0)
    mean_c_b  = np.mean(eval_clip_b, axis=0)
    print(f"{'Dataset Mean (15 img)':<25} | {'V >= 0.95':>10} | {'V >= 0.98':>10} | {'V == 1.0':>12} | {'Any RGB >= 0.98'}")
    print("-" * 75)
    print(f"{'Ground Truth':<25} | {mean_c_gt[0]:>9.2f}% | {mean_c_gt[1]:>9.2f}% | {mean_c_gt[2]:>11.2f}% | {mean_c_gt[3]:>14.2f}%")
    print(f"{'Baseline A (SmoothL1)':<25} | {mean_c_a[0]:>9.2f}% | {mean_c_a[1]:>9.2f}% | {mean_c_a[2]:>11.2f}% | {mean_c_a[3]:>14.2f}%")
    print(f"{'Config B (+HSV Loss)':<25} | {mean_c_b[0]:>9.2f}% | {mean_c_b[1]:>9.2f}% | {mean_c_b[2]:>11.2f}% | {mean_c_b[3]:>14.2f}%")

    # =========================================================================
    # CREATE SUMMARY PUBLICATION-READY FIGURE
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor('#0d1117')

    # Subplot 1: Delta H Boxplot / Violin under different thresholds
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    box_data_a = [delta_h_a[m] for _, m in filters[:5]]
    box_data_b = [delta_h_b[m] for _, m in filters[:5]]
    labels = ["All", "S,V>0.05", "S>0.1\nV>0.05", "S,V>0.1", "S>0.2\nV>0.1"]
    x = np.arange(len(labels))
    w = 0.35
    means_a = [np.mean(d) for d in box_data_a]
    means_b = [np.mean(d) for d in box_data_b]
    rects1 = ax1.bar(x - w/2, means_a, w, label='Baseline A', color='#ff4d4d', alpha=0.85)
    rects2 = ax1.bar(x + w/2, means_b, w, label='Config B (+HSV)', color='#00e676', alpha=0.85)
    ax1.set_ylabel("Mean Shortest Hue Error ΔH (°)", color='white', fontsize=11)
    ax1.set_title("Check 1: ΔH Robustness Across S & V Filters\n(Excluding achromatic near-black noise)", color='white', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax1.grid(True, linestyle=':', alpha=0.3, color='gray')

    # Subplot 2: Saturation KDE / Histogram strictly inside Neon Mask
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    ax2.hist(s_gt_neon, bins=35, range=(0, 1), density=True, alpha=0.55, color='#00b0ff', label=f'Ground Truth (μ={np.mean(s_gt_neon):.2f})')
    ax2.hist(s_a_neon,  bins=35, range=(0, 1), density=True, alpha=0.55, color='#ff4d4d', label=f'Baseline A (μ={np.mean(s_a_neon):.2f})')
    ax2.hist(s_b_neon,  bins=35, range=(0, 1), density=True, alpha=0.55, color='#00e676', label=f'Config B (μ={np.mean(s_b_neon):.2f})')
    ax2.set_xlabel("Saturation S", color='white', fontsize=11)
    ax2.set_ylabel("Probability Density", color='white', fontsize=11)
    ax2.set_title("Check 2: Saturation Distribution in Neon Mask\n(Strict ROI, 31k pixels)", color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax2.grid(True, linestyle=':', alpha=0.3, color='gray')

    # Subplot 3: Clipping vs Gradient in Highlights
    ax3 = axes[2]
    ax3.set_facecolor('#161b22')
    methods = ['Ground Truth', 'Baseline A', 'Config B']
    clips = [c_gt[0], c_a[0], c_b[0]] # V >= 0.95
    grads = [np.mean(grad_gt[hl_mask])*100, np.mean(grad_a[hl_mask])*100, np.mean(grad_b[hl_mask])*100]

    x3 = np.arange(len(methods))
    w3 = 0.35
    ax3.bar(x3 - w3/2, clips, w3, label='Pixel Ratio V ≥ 0.95 (%)', color='#ff9100', alpha=0.85)
    ax3.bar(x3 + w3/2, grads, w3, label='Gradient Magnitude (×100)', color='#9d4edd', alpha=0.85)
    ax3.set_ylabel("Percentage / Magnitude", color='white', fontsize=11)
    ax3.set_title("Check 3: Overexposure vs Gradient Detail\n(Are highlights clipped or texture lost?)", color='white', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(methods, color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#0d1117', edgecolor='none', labelcolor='white')
    ax3.grid(True, linestyle=':', alpha=0.3, color='gray')

    plt.suptitle("Rigorous Pre-Paper Verification of Color & Exposure Claims (LOL eval15)",
                 color='white', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_p = os.path.join(FIGURES_DIR, "pre_paper_verification_three_checks.png")
    out_art = os.path.join(ARTIFACTS_DIR, "pre_paper_verification_three_checks.png")
    plt.savefig(out_p, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    import shutil
    shutil.copy2(out_p, out_art)
    plt.close()
    print(f"\n--> Saved figure: {out_p}")

if __name__ == '__main__':
    main()
