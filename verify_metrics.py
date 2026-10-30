import json
import numpy as np

with open(r'c:\Users\Admin\Documents\LLFormer\ablation_results\ablation_study_abcd_results.json') as f:
    d = json.load(f)

print("=== 1. VERIFY SUMMARY vs PER-IMAGE ARITHMETIC MEAN ===")
for c in ['A_Baseline', 'B_HSV_Loss', 'C_Baseline_CLAHE', 'D_HSV_CLAHE']:
    per_img = d['per_image'][c]
    p_mean = np.mean([r['psnr'] for r in per_img])
    s_mean = np.mean([r['ssim'] for r in per_img])
    e_mean = np.mean([r['delta_e00'] for r in per_img])
    h_mean = np.mean([r['delta_h'] for r in per_img])
    t_mean = np.mean([r['time_ms'] for r in per_img])

    sum_row = [s for s in d['summary'] if s['config_id'] == c][0]
    print(f"\nConfig: {c}")
    print(f"  PSNR:     Per-img mean = {p_mean:.6f} | Summary = {sum_row['mean_psnr']:.6f} | Match: {np.isclose(p_mean, sum_row['mean_psnr'])}")
    print(f"  SSIM:     Per-img mean = {s_mean:.6f} | Summary = {sum_row['mean_ssim']:.6f} | Match: {np.isclose(s_mean, sum_row['mean_ssim'])}")
    print(f"  DeltaE00: Per-img mean = {e_mean:.6f} | Summary = {sum_row['mean_de00']:.6f} | Match: {np.isclose(e_mean, sum_row['mean_de00'])}")
    print(f"  DeltaH:   Per-img mean = {h_mean:.6f} | Summary = {sum_row['mean_dh']:.6f} | Match: {np.isclose(h_mean, sum_row['mean_dh'])}")
    print(f"  Latency:  Per-img mean = {t_mean:.6f} | Summary = {sum_row['mean_time_ms']:.6f} | Match: {np.isclose(t_mean, sum_row['mean_time_ms'])}")

print("\n=== 2. COMPLETE 15-IMAGE RAW DATA TABLE ===")
files = [r['name'] for r in d['per_image']['A_Baseline']]
print(f"{'File':<10} | {'PSNR A':>7} {'PSNR B':>7} | {'SSIM A':>7} {'SSIM B':>7} | {'DE00 A':>7} {'DE00 B':>7} | {'DH A':>7} {'DH B':>7}")
print("-" * 75)
for i, f in enumerate(files):
    rA = d['per_image']['A_Baseline'][i]
    rB = d['per_image']['B_HSV_Loss'][i]
    print(f"{f:<10} | {rA['psnr']:>7.2f} {rB['psnr']:>7.2f} | {rA['ssim']:>7.4f} {rB['ssim']:>7.4f} | {rA['delta_e00']:>7.2f} {rB['delta_e00']:>7.2f} | {rA['delta_h']:>7.2f} {rB['delta_h']:>7.2f}")
