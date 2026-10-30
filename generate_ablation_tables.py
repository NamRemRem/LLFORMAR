"""
Ablation Summary Table Generator
=================================
Tạo bảng số liệu và so sánh Baseline A vs Config B (và các config tiếp theo)
khi training hoàn thành. KHÔNG ghi đè figure cũ.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ABLATION_ROOT = r"c:\Users\Admin\Documents\LLFormer\ablation_results"
FIGURES_DIR = r"c:\Users\Admin\Documents\LLFormer\figures"
ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"
TRAINING_A_JSON = r"c:\Users\Admin\Documents\LLFormer\training_50epochs_results\training_progress.json"

# ─────────────────────────────────────────────────────────────
# 1. Load all available metrics
# ─────────────────────────────────────────────────────────────
def load_config_metrics(config_dir, metrics_filename):
    p = os.path.join(config_dir, metrics_filename)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def load_training_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────────
# 2. Print Console Tables
# ─────────────────────────────────────────────────────────────
def print_summary_table(configs_data):
    """Print markdown-style comparison table"""
    print("\n" + "=" * 80)
    print("   ABLATION STUDY — BẢNG KẾT QUẢ TỔNG HỢP (LOL eval15, 15 ảnh)")
    print("=" * 80)
    print(f"{'Cấu hình':<28} | {'Loss':18s} | {'PSNR (dB)':>9} | {'SSIM':>8} | {'ΔE00 (↓)':>9}")
    print("-" * 80)
    for c in configs_data:
        print(f"{c['label']:<28} | {c['loss_desc']:18s} | {c['psnr']:>9.2f} | {c['ssim']:>8.4f} | {c['de00']:>9.2f}")
    print("=" * 80)
    print("  ΔE00: CIEDE2000 color difference — thấp hơn = màu sắc gần Ground Truth hơn\n")

def print_per_image_table(metrics_a, metrics_b=None):
    """Print per-image breakdown with Δ columns"""
    print("\n" + "=" * 100)
    print("   BẢNG CHI TIẾT TỪNG ẢNH — eval15 (15 ảnh)")
    print("=" * 100)

    header = f"{'Ảnh':8s} | {'PSNR_A':>7} | {'SSIM_A':>7} | {'ΔE00_A':>7}"
    if metrics_b:
        header += f" | {'PSNR_B':>7} | {'SSIM_B':>7} | {'ΔE00_B':>7} | {'ΔPSNR':>7} | {'ΔSSIM':>7} | {'ΔΔE00':>7}"
    print(header)
    print("-" * (100 if metrics_b else 50))

    for r_a in metrics_a['per_image_metrics']:
        fn = r_a['filename']
        line = f"{fn:8s} | {r_a['psnr']:>7.2f} | {r_a['ssim']:>7.4f} | {r_a['delta_e00']:>7.2f}"

        if metrics_b:
            r_b = next((r for r in metrics_b['per_image_metrics'] if r['filename'] == fn), None)
            if r_b:
                d_psnr = r_b['psnr'] - r_a['psnr']
                d_ssim = r_b['ssim'] - r_a['ssim']
                d_de00 = r_b['delta_e00'] - r_a['delta_e00']   # negative = improvement in color
                psnr_s = f"{r_b['psnr']:>7.2f}"
                ssim_s = f"{r_b['ssim']:>7.4f}"
                de_s   = f"{r_b['delta_e00']:>7.2f}"
                dp_s   = f"{'+' if d_psnr >= 0 else ''}{d_psnr:+.2f}"
                ds_s   = f"{'+' if d_ssim >= 0 else ''}{d_ssim:+.4f}"
                dd_s   = f"{'+' if d_de00 >= 0 else ''}{d_de00:+.2f}"
                line += f" | {psnr_s} | {ssim_s} | {de_s} | {dp_s:>7} | {ds_s:>7} | {dd_s:>7}"
            else:
                line += " | (chưa có)"

        print(line)

    print("=" * (100 if metrics_b else 50))
    if metrics_b:
        print(f"{'MEAN':8s} | {metrics_a['mean_psnr']:>7.2f} | {metrics_a['mean_ssim']:>7.4f} | {metrics_a['mean_delta_e00']:>7.2f} | {metrics_b['mean_psnr']:>7.2f} | {metrics_b['mean_ssim']:>7.4f} | {metrics_b['mean_delta_e00']:>7.2f} | {metrics_b['mean_psnr']-metrics_a['mean_psnr']:>+7.2f} | {metrics_b['mean_ssim']-metrics_a['mean_ssim']:>+7.4f} | {metrics_b['mean_delta_e00']-metrics_a['mean_delta_e00']:>+7.2f}")
    else:
        print(f"{'MEAN':8s} | {metrics_a['mean_psnr']:>7.2f} | {metrics_a['mean_ssim']:>7.4f} | {metrics_a['mean_delta_e00']:>7.2f}")
    print()

# ─────────────────────────────────────────────────────────────
# 3. Figure: Grouped Bar Chart Comparison (NEW FILE, no overwrite)
# ─────────────────────────────────────────────────────────────
def generate_ablation_comparison_figure(configs_data, out_prefix="ablation_comparison"):
    """Generate a grouped bar chart and per-image scatter — saved as NEW files"""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), facecolor='#ffffff')
    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.88, bottom=0.12)

    labels = [c['label_short'] for c in configs_data]
    psnrs  = [c['psnr'] for c in configs_data]
    ssims  = [c['ssim'] for c in configs_data]
    de00s  = [c['de00'] for c in configs_data]

    colors = ['#1565c0', '#c0392b', '#2e7d32', '#7b1fa2'][:len(configs_data)]
    x = np.arange(len(labels))
    bw = 0.55

    # PSNR
    ax = axes[0]
    ax.set_facecolor('#ffffff')
    bars = ax.bar(x, psnrs, color=colors, width=bw, edgecolor='#333333', linewidth=0.7)
    for bar, val in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111111')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('PSNR (dB)', fontsize=10.5, fontweight='bold')
    ax.set_title('(A) PSNR Comparison', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(0, max(psnrs) + 2.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')

    # SSIM
    ax = axes[1]
    ax.set_facecolor('#ffffff')
    bars = ax.bar(x, ssims, color=colors, width=bw, edgecolor='#333333', linewidth=0.7)
    for bar, val in zip(bars, ssims):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111111')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('SSIM', fontsize=10.5, fontweight='bold')
    ax.set_title('(B) SSIM Comparison', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(0, min(1.0, max(ssims) + 0.08))
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')

    # ΔE00 (lower = better color accuracy)
    ax = axes[2]
    ax.set_facecolor('#ffffff')
    bars = ax.bar(x, de00s, color=colors, width=bw, edgecolor='#333333', linewidth=0.7)
    for bar, val in zip(bars, de00s):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111111')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('CIEDE2000 ΔE00 (↓ better)', fontsize=10.5, fontweight='bold')
    ax.set_title('(C) Color Accuracy ΔE00', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(0, max(de00s) + 2.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')
    ax.annotate('Lower is better', xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5, color='#555555', style='italic')

    plt.suptitle('Ablation Study Results: LLFormer Configurations on LOL eval15',
                 fontsize=12.5, fontweight='bold', color='#111111', y=0.98)

    for d in [FIGURES_DIR, ARTIFACTS_DIR]:
        out = os.path.join(d, f"{out_prefix}.png")
        plt.savefig(out, dpi=250, facecolor='#ffffff', bbox_inches='tight')
    plt.close()
    print(f"--> Saved: {out_prefix}.png (in figures/ and artifacts/)")

def generate_per_image_scatter(metrics_a, metrics_b, out_prefix="ablation_per_image_scatter"):
    """Scatter of Baseline A ΔE00 vs Config B ΔE00 per image — helps see which images improved"""
    filenames = [r['filename'] for r in metrics_a['per_image_metrics']]
    de_a = [r['delta_e00'] for r in metrics_a['per_image_metrics']]
    de_b_map = {r['filename']: r['delta_e00'] for r in metrics_b['per_image_metrics']}
    de_b = [de_b_map.get(fn, np.nan) for fn in filenames]

    fig, ax = plt.subplots(figsize=(7.5, 6.0), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')

    max_val = max(max(de_a), max([v for v in de_b if not np.isnan(v)])) + 2
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.2, label='No change (y=x)')
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.06, color='green',
                    label='Config B improved (↓ ΔE00)')
    ax.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.06, color='red',
                    label='Config B worsened (↑ ΔE00)')

    for fn, da, db in zip(filenames, de_a, de_b):
        color = '#2e7d32' if db < da else '#c0392b'
        ax.scatter(da, db, s=70, color=color, zorder=5)
        ax.annotate(fn.replace('.png', ''), (da, db), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color='#333333')

    ax.set_xlabel('Baseline A — ΔE00 CIEDE2000', fontsize=11, fontweight='bold')
    ax.set_ylabel('Config B (+ HSV Loss) — ΔE00 CIEDE2000', fontsize=11, fontweight='bold')
    ax.set_title('Per-Image Color Accuracy: Baseline A vs Config B\n(Below diagonal = improvement in color fidelity)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.grid(True, linestyle='--', alpha=0.4, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')
    ax.legend(fontsize=9, loc='upper left')

    for d in [FIGURES_DIR, ARTIFACTS_DIR]:
        out = os.path.join(d, f"{out_prefix}.png")
        plt.savefig(out, dpi=250, facecolor='#ffffff', bbox_inches='tight')
    plt.close()
    print(f"--> Saved: {out_prefix}.png")

def generate_training_curves_comparison(train_a, train_b, out_prefix="ablation_training_curves_AB"):
    """Compare A vs B training loss + PSNR curves (new file, does not touch baseline curves)"""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), facecolor='#ffffff')
    plt.subplots_adjust(wspace=0.28, left=0.08, right=0.97, top=0.88, bottom=0.12)

    ep_a = train_a['epoch']
    ep_b = train_b['epoch']

    # Loss curves
    ax = axes[0]
    ax.set_facecolor('#ffffff')
    l1_a = train_a.get('loss_l1', train_a['loss'])  # fallback to total loss for baseline
    ax.plot(ep_a, l1_a, color='#1565c0', linewidth=2.0, marker='o', markersize=2.5,
            label='A: Baseline (SmoothL1 only)')
    ax.plot(ep_b, train_b['loss_l1'], color='#c0392b', linewidth=2.0, marker='s', markersize=2.5,
            label='B: +HSV Loss (L1 component)')
    ax.set_xlabel('Epoch', fontsize=10.5); ax.set_ylabel('SmoothL1 Loss', fontsize=10.5)
    ax.set_title('(A) SmoothL1 Loss Component', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, linestyle='--', alpha=0.45, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')
    ax.legend(fontsize=9)

    # PSNR curves
    ax = axes[1]
    ax.set_facecolor('#ffffff')
    ax.plot(ep_a, train_a['val_psnr'], color='#1565c0', linewidth=2.0, marker='o', markersize=2.5,
            label='A: Baseline (SmoothL1 only)')
    ax.plot(ep_b, train_b['val_psnr'], color='#c0392b', linewidth=2.0, marker='s', markersize=2.5,
            label='B: +HSV Loss')
    ax.set_xlabel('Epoch', fontsize=10.5); ax.set_ylabel('Val PSNR (dB)', fontsize=10.5)
    ax.set_title('(B) Validation PSNR Progression', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, linestyle='--', alpha=0.45, color='#dcdcdc')
    for spine in ax.spines.values(): spine.set_color('#bbbbbb')
    ax.legend(fontsize=9, loc='lower right')

    plt.suptitle('Ablation: Training Dynamics Comparison — Config A vs Config B',
                 fontsize=12, fontweight='bold', color='#111111', y=0.98)

    for d in [FIGURES_DIR, ARTIFACTS_DIR]:
        out = os.path.join(d, f"{out_prefix}.png")
        plt.savefig(out, dpi=250, facecolor='#ffffff', bbox_inches='tight')
    plt.close()
    print(f"--> Saved: {out_prefix}.png")

# ─────────────────────────────────────────────────────────────
# 4. Main: Generate all tables and figures
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 80)
    print("   ABLATION TABLE & FIGURE GENERATOR")
    print("   (Không ghi đè figure cũ — tất cả output đều là file mới)")
    print("=" * 80)

    # Load Baseline A
    metrics_a = load_config_metrics(
        os.path.join(ABLATION_ROOT, "config_A_baseline"),
        "metrics_baseline_a.json"
    )
    train_a_full = load_training_json(TRAINING_A_JSON)

    # Load Config B (nếu đã xong)
    metrics_b = load_config_metrics(
        os.path.join(ABLATION_ROOT, "config_B_hsv"),
        "metrics_config_b.json"
    )
    train_b_full = load_training_json(
        os.path.join(ABLATION_ROOT, "config_B_hsv", "training_progress.json")
    )

    if metrics_a is None:
        print("ERROR: Baseline A metrics không tìm thấy. Hãy chạy run_ablation_study.py trước.")
        return

    # Build configs_data list
    configs_data = [{
        'label': 'A — Baseline (SmoothL1)',
        'label_short': 'A: Baseline',
        'loss_desc': 'SmoothL1 (RGB)',
        'psnr': metrics_a['mean_psnr'],
        'ssim': metrics_a['mean_ssim'],
        'de00': metrics_a['mean_delta_e00'],
    }]

    if metrics_b:
        configs_data.append({
            'label': 'B — Baseline + HSV Loss',
            'label_short': 'B: +HSV Loss',
            'loss_desc': 'SmoothL1 + HSV (λ=0.15)',
            'psnr': metrics_b['mean_psnr'],
            'ssim': metrics_b['mean_ssim'],
            'de00': metrics_b['mean_delta_e00'],
        })

    # ── 1. Console Tables ──
    print_summary_table(configs_data)
    if metrics_b:
        print_per_image_table(metrics_a, metrics_b)
    else:
        print_per_image_table(metrics_a, None)
        print("\n[!] Config B chưa hoàn thành — chỉ hiển thị Baseline A.\n"
              "    Chạy lại generate_ablation_tables.py sau khi training xong.\n")

    # ── 2. Bar Chart Figure (NEW file) ──
    out_prefix = "ablation_bar_comparison_A"
    if metrics_b:
        out_prefix = "ablation_bar_comparison_AB"
    generate_ablation_comparison_figure(configs_data, out_prefix=out_prefix)

    # ── 3. Per-image ΔE00 scatter + training curve comparison ──
    if metrics_b and train_b_full:
        generate_per_image_scatter(metrics_a, metrics_b,
                                   out_prefix="ablation_per_image_scatter_AB")
        if train_a_full and train_b_full:
            generate_training_curves_comparison(train_a_full, train_b_full,
                                                out_prefix="ablation_training_curves_AB")

    print("\n[DONE] Tất cả bảng số liệu và figure đã được xuất thành công!")
    print(f"  → figures/ablation_*.png   (bộ figure mới, không thay thế file cũ)")
    print(f"  → Bảng text in trực tiếp ra terminal ở trên")

if __name__ == '__main__':
    main()
