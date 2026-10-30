"""
Complete Ablation Study: 4-Configuration Experiment (A, B, C, D)
================================================================
A: LLFormer Base (SmoothL1)
B: LLFormer Base + HSV Loss (lambda=0.15)
C: LLFormer Base + Histogram Post-processing (CLAHE on V-channel)
D: LLFormer Base + HSV Loss + Histogram Post-processing (CLAHE on V-channel)

Strict Controls:
- Same test dataset: LOL eval15 (15 image pairs)
- Fixed histogram algorithm: CLAHE on Value (V) in HSV, clipLimit=1.5, tileGrid=(8,8)
- Zero leakage: No test ground truth used for histogram tuning
- Comprehensive metrics: PSNR, SSIM, DeltaE00, Circular Hue Error Delta H,
  Neon-pink Saturation, Neon-pink Hue Retention, and Inference Time (ms/img).
"""
import os
import sys
import time
import json
import shutil
import cv2
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
ABLATION_DIR = r"c:\Users\Admin\Documents\LLFormer\ablation_results"

CKPT_A = r"c:\Users\Admin\Documents\LLFormer\training_50epochs_results\checkpoints\model_bestPSNR.pth"
CKPT_B = r"c:\Users\Admin\Documents\LLFormer\ablation_results\config_B_hsv\checkpoints\model_bestPSNR.pth"

# Fixed CLAHE hyperparameters (chosen without touching test GT)
CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)
clahe_engine = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

def shortest_circular_hue_diff_deg(h1_deg, h2_deg):
    diff = np.abs(h1_deg - h2_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)

def calc_delta_e00(pred_rgb, gt_rgb):
    pred_lab = color.rgb2lab(pred_rgb)
    gt_lab   = color.rgb2lab(gt_rgb)
    de00     = color.deltaE_ciede2000(pred_lab, gt_lab)
    return float(np.mean(de00)), de00

def apply_clahe_on_v(rgb_img_float):
    """
    Applies CLAHE strictly on the Value (V) channel in HSV space.
    H (Hue) and S (Saturation) are strictly preserved (100% untouched).
    """
    hsv = color.rgb2hsv(rgb_img_float)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    v_u8 = np.clip(v_chan * 255.0, 0, 255).astype(np.uint8)
    v_eq_u8 = clahe_engine.apply(v_u8)
    v_eq = v_eq_u8.astype(np.float32) / 255.0

    hsv_new = np.stack([h_chan, s_chan, v_eq], axis=-1)
    rgb_new = color.hsv2rgb(hsv_new).astype(np.float32)
    return np.clip(rgb_new, 0.0, 1.0)

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

def predict_gpu(model, t_l, device):
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
    print("=" * 80)
    print(f"   RUNNING COMPLETE 4-CONFIGURATION ABLATION STUDY (A, B, C, D)")
    print(f"   Device: {device}")
    print(f"   Histogram Strategy: CLAHE on HSV V-channel (clipLimit={CLAHE_CLIP}, grid={CLAHE_GRID})")
    print("=" * 80)

    print("--> Loading Baseline Model A...")
    model_a = load_model(CKPT_A, device)
    print("--> Loading Config B (+HSV Loss) Model...")
    model_b = load_model(CKPT_B, device)

    # Warmup GPU
    dummy = torch.zeros(1, 3, 256, 256).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model_a(dummy); _ = model_b(dummy)
    torch.cuda.synchronize()

    eval_files = sorted([f for f in os.listdir(os.path.join(EVAL_DIR, 'low')) if f.endswith(('.png','.jpg'))])
    print(f"--> Found {len(eval_files)} images in eval15.")

    configs = ['A_Baseline', 'B_HSV_Loss', 'C_Baseline_CLAHE', 'D_HSV_CLAHE']
    config_labels = {
        'A_Baseline': 'A: Baseline (SmoothL1)',
        'B_HSV_Loss': 'B: Base + HSV Loss',
        'C_Baseline_CLAHE': 'C: Base + Histogram (CLAHE)',
        'D_HSV_CLAHE': 'D: Base + HSV + Histogram'
    }

    results_per_config = {c: [] for c in configs}
    timing_per_config  = {c: [] for c in configs}

    # Store full outputs for visual gallery
    saved_outputs = {}

    for idx, fname in enumerate(eval_files):
        low_p = os.path.join(EVAL_DIR, 'low', fname)
        high_p = os.path.join(EVAL_DIR, 'high', fname)

        img_l_pil = Image.open(low_p).convert('RGB')
        img_h_pil = Image.open(high_p).convert('RGB')

        arr_gt = np.array(img_h_pil).astype(np.float32) / 255.0
        arr_low = np.array(img_l_pil).astype(np.float32) / 255.0

        t_l = TF.to_tensor(img_l_pil).unsqueeze(0).to(device)

        # 1. Config A: Baseline
        torch.cuda.synchronize(); t0 = time.time()
        arr_a = predict_gpu(model_a, t_l, device)
        torch.cuda.synchronize(); t_a = (time.time() - t0) * 1000.0

        # 2. Config B: HSV Loss
        torch.cuda.synchronize(); t0 = time.time()
        arr_b = predict_gpu(model_b, t_l, device)
        torch.cuda.synchronize(); t_b = (time.time() - t0) * 1000.0

        # 3. Config C: Baseline + CLAHE
        t0 = time.time()
        arr_c = apply_clahe_on_v(arr_a)
        t_c = t_a + (time.time() - t0) * 1000.0

        # 4. Config D: HSV Loss + CLAHE
        t0 = time.time()
        arr_d = apply_clahe_on_v(arr_b)
        t_d = t_b + (time.time() - t0) * 1000.0

        outputs = {'A_Baseline': arr_a, 'B_HSV_Loss': arr_b, 'C_Baseline_CLAHE': arr_c, 'D_HSV_CLAHE': arr_d}
        times   = {'A_Baseline': t_a,   'B_HSV_Loss': t_b,   'C_Baseline_CLAHE': t_c,   'D_HSV_CLAHE': t_d}

        if fname in ['179.png', '111.png', '23.png', '669.png']:
            saved_outputs[fname] = {
                'low': arr_low, 'gt': arr_gt,
                'A': arr_a, 'B': arr_b, 'C': arr_c, 'D': arr_d
            }

        # Calculate metrics against GT
        hsv_gt = color.rgb2hsv(arr_gt)
        h_gt_deg = hsv_gt[:, :, 0] * 360.0
        valid_hue_mask = (hsv_gt[:, :, 1] > 0.10) & (hsv_gt[:, :, 2] > 0.10)

        for c_key in configs:
            pred = outputs[c_key]
            psnr_val = float(psnr_fn(arr_gt, pred, data_range=1.0))
            ssim_val = float(ssim_fn(arr_gt, pred, data_range=1.0, channel_axis=2))
            de00_val, _ = calc_delta_e00(pred, arr_gt)

            # Circular Hue Error
            hsv_pred = color.rgb2hsv(pred)
            h_pred_deg = hsv_pred[:, :, 0] * 360.0
            dh = shortest_circular_hue_diff_deg(h_pred_deg[valid_hue_mask], h_gt_deg[valid_hue_mask])
            dh_val = float(np.mean(dh))

            results_per_config[c_key].append({
                'name': fname,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'delta_e00': de00_val,
                'delta_h': dh_val,
                'time_ms': times[c_key]
            })
            timing_per_config[c_key].append(times[c_key])

        print(f"[{idx+1:02d}/15] {fname:<10} | "
              f"PSNR(A:{outputs['A_Baseline'] is not None and results_per_config['A_Baseline'][-1]['psnr']:.1f}, "
              f"B:{results_per_config['B_HSV_Loss'][-1]['psnr']:.1f}, "
              f"C:{results_per_config['C_Baseline_CLAHE'][-1]['psnr']:.1f}, "
              f"D:{results_per_config['D_HSV_CLAHE'][-1]['psnr']:.1f}) | "
              f"ΔE00(A:{results_per_config['A_Baseline'][-1]['delta_e00']:.1f}, "
              f"B:{results_per_config['B_HSV_Loss'][-1]['delta_e00']:.1f}, "
              f"C:{results_per_config['C_Baseline_CLAHE'][-1]['delta_e00']:.1f}, "
              f"D:{results_per_config['D_HSV_CLAHE'][-1]['delta_e00']:.1f})")

    # ── Specialized Neon Pink ROI Analysis on 179.png ─────────────────────────
    print("\n--> Evaluating Neon Pink Region Metrics on 179.png...")
    arr_gt_179 = saved_outputs['179.png']['gt']
    hsv_gt_179 = color.rgb2hsv(arr_gt_179)
    H_gt_179 = hsv_gt_179[:, :, 0] * 360.0
    S_gt_179 = hsv_gt_179[:, :, 1]
    V_gt_179 = hsv_gt_179[:, :, 2]

    neon_mask = (H_gt_179 >= 280.0) & (H_gt_179 <= 355.0) & (S_gt_179 >= 0.35) & (V_gt_179 >= 0.15)
    # Simple key mapping config_id -> saved_outputs key
    CKEY_TO_SAVED = {
        'A_Baseline':      'A',
        'B_HSV_Loss':      'B',
        'C_Baseline_CLAHE':'C',
        'D_HSV_CLAHE':     'D',
    }

    neon_metrics = {}
    neon_metrics['GT'] = {
        'mean_s': float(np.mean(S_gt_179[neon_mask])),
        'hue_pink_pct': 100.0,
        'hue_yellow_pct': 0.0
    }

    for c_key in configs:
        pred_arr = saved_outputs['179.png'][CKEY_TO_SAVED[c_key]]
        hsv_p = color.rgb2hsv(pred_arr)
        H_p = hsv_p[:, :, 0] * 360.0
        S_p = hsv_p[:, :, 1]

        mean_s = float(np.mean(S_p[neon_mask]))
        pink_pct = float(np.mean((H_p[neon_mask] >= 280.0) | (H_p[neon_mask] <= 20.0)) * 100.0)
        yellow_pct = float(np.mean((H_p[neon_mask] >= 35.0) & (H_p[neon_mask] <= 75.0)) * 100.0)

        neon_metrics[c_key] = {
            'mean_s': mean_s,
            'recovery_s_pct': (mean_s / neon_metrics['GT']['mean_s']) * 100.0,
            'hue_pink_pct': pink_pct,
            'hue_yellow_pct': yellow_pct
        }

    # ── Summary Metrics Computation ──────────────────────────────────────────
    summary_table = []
    for c_key in configs:
        res = results_per_config[c_key]
        summary_table.append({
            'config_id': c_key,
            'label': config_labels[c_key],
            'mean_psnr': float(np.mean([r['psnr'] for r in res])),
            'mean_ssim': float(np.mean([r['ssim'] for r in res])),
            'mean_de00': float(np.mean([r['delta_e00'] for r in res])),
            'mean_dh': float(np.mean([r['delta_h'] for r in res])),
            'mean_time_ms': float(np.mean(timing_per_config[c_key])),
            'neon_mean_s': neon_metrics[c_key]['mean_s'],
            'neon_rec_pct': neon_metrics[c_key]['recovery_s_pct'],
            'neon_pink_pct': neon_metrics[c_key]['hue_pink_pct'],
            'neon_yellow_pct': neon_metrics[c_key]['hue_yellow_pct']
        })

    # Save to JSON
    full_ablation_data = {
        'summary': summary_table,
        'per_image': results_per_config,
        'neon_analysis': neon_metrics,
        'clahe_config': {'clipLimit': CLAHE_CLIP, 'tileGridSize': CLAHE_GRID, 'channel': 'HSV-Value'}
    }
    with open(os.path.join(ABLATION_DIR, 'ablation_study_abcd_results.json'), 'w') as f:
        json.dump(full_ablation_data, f, indent=2)

    # ── Console Presentation ─────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print("   ABLATION STUDY — BẢNG KẾT QUẢ TỔNG HỢP 4 CẤU HÌNH (LOL eval15)")
    print("=" * 105)
    print(f"{'Cấu hình':<30} | {'PSNR (dB)':>9} | {'SSIM':>8} | {'ΔE00 (↓)':>9} | {'ΔH (°)(↓)':>10} | {'Neon Sat':>8} | {'Pink Ret%':>9} | {'Latency':>8}")
    print("-" * 105)
    for row in summary_table:
        print(f"{row['label']:<30} | {row['mean_psnr']:>9.2f} | {row['mean_ssim']:>8.4f} | {row['mean_de00']:>9.2f} | "
              f"{row['mean_dh']:>10.2f}° | {row['neon_mean_s']:>8.3f} | {row['neon_pink_pct']:>8.1f}% | {row['mean_time_ms']:>6.1f}ms")
    print("=" * 105)
    print(f"Ground Truth Reference (Neon 179): Mean Sat = {neon_metrics['GT']['mean_s']:.3f} | Pink = 100.0% | Yellow = 0.0%\n")

    # ── Per-Image Comparison Table ───────────────────────────────────────────
    print("=" * 115)
    print("   CHI TIẾT PSNR & ΔE00 TRÊN TỪNG ẢNH (eval15)")
    print("=" * 115)
    print(f"{'Image':<8} | {'PSNR A':>7} | {'PSNR B':>7} | {'PSNR C':>7} | {'PSNR D':>7} | "
          f"{'ΔE00 A':>7} | {'ΔE00 B':>7} | {'ΔE00 C':>7} | {'ΔE00 D':>7} | {'Best Config'}")
    print("-" * 115)
    for idx, fname in enumerate(eval_files):
        pA = results_per_config['A_Baseline'][idx]['psnr']
        pB = results_per_config['B_HSV_Loss'][idx]['psnr']
        pC = results_per_config['C_Baseline_CLAHE'][idx]['psnr']
        pD = results_per_config['D_HSV_CLAHE'][idx]['psnr']

        eA = results_per_config['A_Baseline'][idx]['delta_e00']
        eB = results_per_config['B_HSV_Loss'][idx]['delta_e00']
        eC = results_per_config['C_Baseline_CLAHE'][idx]['delta_e00']
        eD = results_per_config['D_HSV_CLAHE'][idx]['delta_e00']

        best_p = ['A', 'B', 'C', 'D'][np.argmax([pA, pB, pC, pD])]
        best_e = ['A', 'B', 'C', 'D'][np.argmin([eA, eB, eC, eD])]
        print(f"{fname:<8} | {pA:>7.2f} | {pB:>7.2f} | {pC:>7.2f} | {pD:>7.2f} | "
              f"{eA:>7.2f} | {eB:>7.2f} | {eC:>7.2f} | {eD:>7.2f} | PSNR:{best_p}, Color:{best_e}")
    print("=" * 115)

    # ── PLOT 1: Summary Bar Chart (PSNR, SSIM, DeltaE00, DeltaH) ─────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor('#0d1117')

    cfg_names = ['A: Base', 'B: +HSV', 'C: +Hist', 'D: +HSV+Hist']
    cfg_colors = ['#38bdf8', '#00e676', '#f59e0b', '#ec4899']

    # PSNR
    ax = axes[0]; ax.set_facecolor('#161b22')
    vals = [r['mean_psnr'] for r in summary_table]
    bars = ax.bar(cfg_names, vals, color=cfg_colors, alpha=0.9, width=0.55)
    ax.set_title("PSNR (dB) ↑ [Pixel Reconstruction]", color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{b.get_height():.2f}",
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

    # SSIM
    ax = axes[1]; ax.set_facecolor('#161b22')
    vals = [r['mean_ssim'] for r in summary_table]
    bars = ax.bar(cfg_names, vals, color=cfg_colors, alpha=0.9, width=0.55)
    ax.set_title("SSIM ↑ [Structural Similarity]", color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.4f}",
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

    # Delta E00
    ax = axes[2]; ax.set_facecolor('#161b22')
    vals = [r['mean_de00'] for r in summary_table]
    bars = ax.bar(cfg_names, vals, color=cfg_colors, alpha=0.9, width=0.55)
    ax.set_title("ΔE₀₀ (CIEDE2000) ↓ [Perceptual Color Error]", color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{b.get_height():.2f}",
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

    # Delta H
    ax = axes[3]; ax.set_facecolor('#161b22')
    vals = [r['mean_dh'] for r in summary_table]
    bars = ax.bar(cfg_names, vals, color=cfg_colors, alpha=0.9, width=0.55)
    ax.set_title("Circular Hue Error ΔH (°) ↓ [Color Tone]", color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.2f}°",
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

    plt.suptitle("Ablation Study: Quantitative Comparison Across Configurations A, B, C, D (LOL eval15)",
                 color='white', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    bar_p = os.path.join(FIGURES_DIR, "ablation_abcd_metrics_bar.png")
    plt.savefig(bar_p, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    shutil.copy2(bar_p, os.path.join(ARTIFACTS_DIR, "ablation_abcd_metrics_bar.png"))
    plt.close()
    print("--> Saved figure: ablation_abcd_metrics_bar.png")

    # ── PLOT 2: Comprehensive Visual Gallery (6 Columns: Input, A, B, C, D, GT) ──
    print("--> Generating 6-Column Visual Gallery for key test scenes...")
    gallery_scenes = ['111.png', '179.png', '23.png', '669.png']
    fig, axes = plt.subplots(len(gallery_scenes), 6, figsize=(28, 4.5 * len(gallery_scenes)))
    fig.patch.set_facecolor('#0d1117')

    col_headers = [
        "Low-Light Input",
        "Config A: Baseline",
        "Config B: +HSV Loss",
        "Config C: +Histogram",
        "Config D: +HSV + Histogram",
        "Ground Truth (Reference)"
    ]

    for row_idx, sname in enumerate(gallery_scenes):
        sdata = saved_outputs[sname]
        imgs_row = [sdata['low'], sdata['A'], sdata['B'], sdata['C'], sdata['D'], sdata['gt']]

        # Get metrics for labels
        idx_in_eval = eval_files.index(sname)
        m_a = results_per_config['A_Baseline'][idx_in_eval]
        m_b = results_per_config['B_HSV_Loss'][idx_in_eval]
        m_c = results_per_config['C_Baseline_CLAHE'][idx_in_eval]
        m_d = results_per_config['D_HSV_CLAHE'][idx_in_eval]

        labels_row = [
            f"{sname} (Input)",
            f"PSNR:{m_a['psnr']:.1f}dB | ΔE₀₀:{m_a['delta_e00']:.1f}",
            f"PSNR:{m_b['psnr']:.1f}dB | ΔE₀₀:{m_b['delta_e00']:.1f}",
            f"PSNR:{m_c['psnr']:.1f}dB | ΔE₀₀:{m_c['delta_e00']:.1f}",
            f"PSNR:{m_d['psnr']:.1f}dB | ΔE₀₀:{m_d['delta_e00']:.1f}",
            "Ground Truth"
        ]

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs_row[col_idx])
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], color='white', fontsize=13, fontweight='bold', pad=12)

            color_tag = 'white' if col_idx in [0, 5] else cfg_colors[col_idx - 1]
            ax.text(0.02, 0.04, labels_row[col_idx], transform=ax.transAxes, color=color_tag,
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75), fontsize=10, fontweight='bold')

    plt.suptitle("Visual Comparison Across Ablation Configurations: A (Baseline), B (+HSV), C (+Hist), D (+HSV+Hist)",
                 color='white', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    gal_p = os.path.join(FIGURES_DIR, "ablation_abcd_visual_gallery.png")
    plt.savefig(gal_p, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    shutil.copy2(gal_p, os.path.join(ARTIFACTS_DIR, "ablation_abcd_visual_gallery.png"))
    plt.close()
    print("--> Saved figure: ablation_abcd_visual_gallery.png")

    print("\n[COMPLETE] 4-Configuration Ablation Study Finished Successfully!")

if __name__ == '__main__':
    main()
