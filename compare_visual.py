import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from model.LLFormer import LLFormer

def parse_args():
    parser = argparse.ArgumentParser(description="Matplotlib Comparison: Ground Truth vs LLFormer vs Low-Light")
    parser.add_argument('--image_name', type=str, default='1.png', help='Tên file ảnh trong LOLdataset/eval15 (ví dụ: 1.png, 22.png, 55.png)')
    parser.add_argument('--weights', type=str, default='./preliminary_results/checkpoints/model_bestPSNR.pth', help='Đường dẫn checkpoint')
    parser.add_argument('--save_path', type=str, default='./preliminary_results/comparison_llformer_vs_gt.png', help='Nơi lưu biểu đồ matplotlib')
    parser.add_argument('--show', action='store_true', help='Hiển thị cửa sổ tương tác matplotlib')
    return parser.parse_args()

def main():
    args = parse_args()
    
    low_path = os.path.join('./LOLdataset/eval15/low', args.image_name)
    high_path = os.path.join('./LOLdataset/eval15/high', args.image_name)
    
    if not os.path.exists(low_path) or not os.path.exists(high_path):
        print(f"Lỗi: Không tìm thấy ảnh {args.image_name} trong eval15.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang tải mô hình LLFormer lên {device}...")

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

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load images
    raw_low_img = Image.open(low_path).convert('RGB')
    raw_high_img = Image.open(high_path).convert('RGB')
    
    low_tensor = TF.to_tensor(raw_low_img).unsqueeze(0).to(device)
    h, w = low_tensor.shape[2], low_tensor.shape[3]
    mul = 16
    H, W = ((h + mul - 1) // mul) * mul, ((w + mul - 1) // mul) * mul
    padh, padw = H - h, W - w
    input_padded = F.pad(low_tensor, (0, padw, 0, padh), 'reflect')

    print(f"Đang suy luận ảnh độ phân giải gốc {w}x{h}...")
    t0 = time.time()
    with torch.no_grad():
        restored = model(input_padded)
    infer_time = time.time() - t0
    print(f"Hoàn thành suy luận trong {infer_time:.2f}s!")

    restored = torch.clamp(restored[:, :, :h, :w], 0, 1)
    
    # Chuyển sang numpy uint8 và float [0, 1]
    img_low = np.array(raw_low_img)
    img_gt = np.array(raw_high_img)
    img_pred = (restored[0].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # Tính toán số liệu PSNR và SSIM
    metric_psnr_in = psnr(img_gt, img_low)
    metric_ssim_in = ssim(img_gt, img_low, channel_axis=2)

    metric_psnr_pred = psnr(img_gt, img_pred)
    metric_ssim_pred = ssim(img_gt, img_pred, channel_axis=2)

    mae_val = np.mean(np.abs(img_gt.astype(float) - img_pred.astype(float)))
    rmse_val = np.sqrt(np.mean((img_gt.astype(float) - img_pred.astype(float)) ** 2))

    # Tọa độ vùng zoom chi tiết (ROI Box)
    box_size = 90
    box_y = int(h * 0.45)
    box_x = int(w * 0.40)
    
    crop_low = img_low[box_y:box_y+box_size, box_x:box_x+box_size]
    crop_pred = img_pred[box_y:box_y+box_size, box_x:box_x+box_size]
    crop_gt = img_gt[box_y:box_y+box_size, box_x:box_x+box_size]

    # Bản đồ sai số tuyệt đối (Error Heatmap)
    diff_map = np.mean(np.abs(img_gt.astype(float) - img_pred.astype(float)), axis=2)

    # Khởi tạo Matplotlib Figure phong cách Dark Theme hiện đại
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12), facecolor='#090d16')
    
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.28, wspace=0.18)

    # ROW 1: Ảnh toàn cảnh (Full Images)
    # 1. Low-Light Input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_low)
    ax1.set_title(f"[1] Low-Light Input (Gốc)\nPSNR: {metric_psnr_in:.2f} dB | SSIM: {metric_ssim_in:.4f}", color='#f43f5e', fontsize=12, fontweight='bold', pad=8)
    rect1 = patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#ef4444', facecolor='none', linestyle='--')
    ax1.add_patch(rect1)
    ax1.axis('off')

    # 2. LLFormer Output
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_pred)
    ax2.set_title(f"[2] LLFormer Enhanced (Phục hồi)\nPSNR: {metric_psnr_pred:.2f} dB | SSIM: {metric_ssim_pred:.4f}", color='#34d399', fontsize=12, fontweight='bold', pad=8)
    rect2 = patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#10b981', facecolor='none', linestyle='--')
    ax2.add_patch(rect2)
    ax2.axis('off')

    # 3. Ground Truth
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_gt)
    ax3.set_title(f"[3] Ground Truth (Ảnh chuẩn đủ sáng)\nTarget Reference", color='#38bdf8', fontsize=12, fontweight='bold', pad=8)
    rect3 = patches.Rectangle((box_x, box_y), box_size, box_size, linewidth=2, edgecolor='#0284c7', facecolor='none', linestyle='--')
    ax3.add_patch(rect3)
    ax3.axis('off')

    # ROW 2: Vùng zoom chi tiết (Zoom-in Region of Interest)
    # 4. Zoom Low
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(crop_low)
    ax4.set_title("Vùng phóng to: Low-Light (Tối đen, mất chi tiết)", color='#fda4af', fontsize=11, pad=6)
    ax4.axis('off')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#ef4444'); spine.set_linewidth(2)

    # 5. Zoom LLFormer
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(crop_pred)
    ax5.set_title("Vùng phóng to: LLFormer (Khôi phục cấu trúc & màu)", color='#6ee7b7', fontsize=11, pad=6)
    ax5.axis('off')
    for spine in ax5.spines.values():
        spine.set_edgecolor('#10b981'); spine.set_linewidth(2)

    # 6. Zoom Ground Truth
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(crop_gt)
    ax6.set_title("Vùng phóng to: Ground Truth (Chi tiết nguyên bản)", color='#7dd3fc', fontsize=11, pad=6)
    ax6.axis('off')
    for spine in ax6.spines.values():
        spine.set_edgecolor('#0284c7'); spine.set_linewidth(2)

    # ROW 3: Đánh giá số liệu & Phân tích
    # 7. Bản đồ sai lệch (Error Heatmap)
    ax7 = fig.add_subplot(gs[2, 0])
    im_err = ax7.imshow(diff_map, cmap='inferno', vmin=0, vmax=100)
    ax7.set_title("Bản đồ sai số tuyệt đối |LLFormer - GT|", color='#e2e8f0', fontsize=11, fontweight='bold', pad=6)
    ax7.axis('off')
    cbar = fig.colorbar(im_err, ax=ax7, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#e2e8f0', labelsize=9)
    cbar.set_label('Chênh lệch giá trị pixel (0-255)', color='#cbd5e1', fontsize=9)

    # 8. Histogram phổ sáng RGB (Luminance Distribution)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_facecolor('#131b2e')
    # Gray luminance
    gray_low = np.mean(img_low, axis=2).ravel()
    gray_pred = np.mean(img_pred, axis=2).ravel()
    gray_gt = np.mean(img_gt, axis=2).ravel()

    ax8.hist(gray_low, bins=50, color='#f43f5e', alpha=0.55, label='Low-Light (Bị nén ở vùng tối)', density=True)
    ax8.hist(gray_pred, bins=50, color='#34d399', alpha=0.55, label='LLFormer (Được kéo đều phổ)', density=True)
    ax8.hist(gray_gt, bins=50, color='#38bdf8', alpha=0.45, histtype='step', linewidth=2, label='Ground Truth (Chuẩn)', density=True)
    ax8.set_title("Biểu đồ phân bố độ sáng (Luminance Histogram)", color='#e2e8f0', fontsize=11, fontweight='bold', pad=8)
    ax8.set_xlabel("Mức độ sáng (0 - 255)", color='#94a3b8', fontsize=9)
    ax8.set_ylabel("Mật độ", color='#94a3b8', fontsize=9)
    ax8.tick_params(colors='#94a3b8', labelsize=8)
    ax8.legend(fontsize=8, loc='upper right', facecolor='#0b1120', edgecolor='#334155', labelcolor='#e2e8f0')
    ax8.grid(True, linestyle=':', alpha=0.3, color='#475569')

    # 9. Bảng đối chiếu định lượng (Metrics Comparison Table)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    table_vals = [
        ['Chỉ số đánh giá', 'Ảnh gốc Low-Light', 'Mô hình LLFormer', 'Mức cải thiện'],
        ['PSNR (dB)', f"{metric_psnr_in:.2f} dB", f"{metric_psnr_pred:.2f} dB", f"+{metric_psnr_pred - metric_psnr_in:.2f} dB"],
        ['SSIM (Cấu trúc)', f"{metric_ssim_in:.4f}", f"{metric_ssim_pred:.4f}", f"+{metric_ssim_pred - metric_ssim_in:.4f}"],
        ['Sai số MAE', f"{np.mean(np.abs(img_gt.astype(float) - img_low.astype(float))):.1f}", f"{mae_val:.1f}", f"Giảm {np.mean(np.abs(img_gt.astype(float) - img_low.astype(float))) - mae_val:.1f}"],
        ['Độ phân giải', f"{w} x {h}", f"{w} x {h}", "Nguyên bản"],
        ['Thời gian suy luận', '-', f"{infer_time:.2f} giây", "RTX 4050 GPU"]
    ]
    tbl = ax9.table(cellText=table_vals, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.05, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#334155')
        if r == 0:
            cell.set_facecolor('#0284c7')
            cell.set_text_props(color='#ffffff', weight='bold')
        else:
            cell.set_facecolor('#131b2e' if r % 2 == 0 else '#1e293b')
            cell.set_text_props(color='#f1f5f9')

    ax9.set_title("Bảng tổng kết chỉ số định lượng chi tiết", color='#e2e8f0', fontsize=11, fontweight='bold', pad=15)

    # Tiêu đề chính
    fig.suptitle(f"SO SÁNH TRỰC QUAN MATPLOTLIB: LOW-LIGHT vs LLFORMER vs GROUND TRUTH (Ảnh: {args.image_name})", 
                 fontsize=15, fontweight='bold', color='#38bdf8', y=0.98)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"--> Đã lưu biểu đồ Matplotlib so sánh chi tiết tại: {args.save_path}")

    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
