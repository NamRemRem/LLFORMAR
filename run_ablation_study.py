import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage.color as color

from model.LLFormer import LLFormer
import utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.backends.cudnn.benchmark = True

BASE_DIR = r"c:\Users\Admin\Documents\LLFormer"
EVAL_DIR = os.path.join(BASE_DIR, "LOLdataset", "eval15")
TRAIN_DIR = os.path.join(BASE_DIR, "LOLdataset", "our485")
ABLATION_ROOT = os.path.join(BASE_DIR, "ablation_results")
os.makedirs(ABLATION_ROOT, exist_ok=True)

# -------------------------------------------------------------
# 1. Differentiable Color Space & Histogram Losses
# -------------------------------------------------------------
def rgb_to_hsv_diff(rgb: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Fully differentiable RGB to HSV conversion without in-place operations"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    cmax, _ = torch.max(rgb, dim=1, keepdim=True)
    cmin, _ = torch.min(rgb, dim=1, keepdim=True)
    delta = cmax - cmin + eps

    r_is_max = (cmax == r).float()
    g_is_max = ((cmax == g) & (cmax != r)).float()
    b_is_max = ((cmax != r) & (cmax != g)).float()

    diff_h = (g - b) / delta
    diff_h = diff_h - 6.0 * torch.floor(diff_h / 6.0)

    h = r_is_max * diff_h + g_is_max * (((b - r) / delta) + 2.0) + b_is_max * (((r - g) / delta) + 4.0)
    h = h / 6.0
    s = (cmax - cmin) / (cmax + eps)
    v = cmax
    return torch.cat([h, s, v], dim=1)

def chroma_hsv_loss(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
    """
    Continuous circular chromatic HSV loss:
    Maps (H, S) to Cartesian chrominance plane (Cx, Cy) to avoid Hue circular discontinuity
    and zero-saturation singularity.
    """
    hsv_p = rgb_to_hsv_diff(rgb_pred)
    hsv_g = rgb_to_hsv_diff(rgb_gt)

    h_p, s_p = hsv_p[:, 0:1], hsv_p[:, 1:2]
    h_g, s_g = hsv_g[:, 0:1], hsv_g[:, 1:2]

    cx_p = s_p * torch.cos(2 * 3.1415926535 * h_p)
    cy_p = s_p * torch.sin(2 * 3.1415926535 * h_p)
    cx_g = s_g * torch.cos(2 * 3.1415926535 * h_g)
    cy_g = s_g * torch.sin(2 * 3.1415926535 * h_g)

    return F.l1_loss(cx_p, cx_g) + F.l1_loss(cy_p, cy_g)

def histogram_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 1D Wasserstein histogram loss:
    L1 distance between sorted pixel values per channel = Exact 1D Earth Mover's Distance.
    """
    b, c, h, w = pred.shape
    p_sorted = torch.sort(pred.view(b, c, -1), dim=-1)[0]
    g_sorted = torch.sort(gt.view(b, c, -1), dim=-1)[0]
    return F.l1_loss(p_sorted, g_sorted)

# -------------------------------------------------------------
# 2. Fast In-RAM Dataset
# -------------------------------------------------------------
class FastRAMDataset(Dataset):
    def __init__(self, rgb_dir, patch_size=128):
        super().__init__()
        self.ps = patch_size
        low_dir = os.path.join(rgb_dir, 'low')
        high_dir = os.path.join(rgb_dir, 'high')

        inp_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        tar_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        self.samples = []
        for f_low, f_high in zip(inp_files, tar_files):
            img_l = TF.to_tensor(Image.open(os.path.join(low_dir, f_low)).convert('RGB'))
            img_h = TF.to_tensor(Image.open(os.path.join(high_dir, f_high)).convert('RGB'))
            self.samples.append((img_h, img_l))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_img, inp_img = self.samples[idx]
        ps = self.ps
        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        crop_tar = tar_img[:, rr:rr + ps, cc:cc + ps]
        crop_inp = inp_img[:, rr:rr + ps, cc:cc + ps]

        aug = random.randint(0, 4)
        if aug == 1:
            crop_inp = crop_inp.flip(1); crop_tar = crop_tar.flip(1)
        elif aug == 2:
            crop_inp = crop_inp.flip(2); crop_tar = crop_tar.flip(2)
        elif aug == 3:
            crop_inp = torch.rot90(crop_inp, 1, [1, 2]); crop_tar = torch.rot90(crop_tar, 1, [1, 2])

        return crop_tar, crop_inp

# -------------------------------------------------------------
# 3. Comprehensive Evaluation: PSNR, SSIM, CIEDE2000 (ΔE00)
# -------------------------------------------------------------
def evaluate_model_on_eval15(model, device):
    model.eval()
    low_files = sorted([f for f in os.listdir(os.path.join(EVAL_DIR, 'low')) if f.endswith(('.png', '.jpg'))])

    results = []
    with torch.no_grad():
        for fname in low_files:
            low_p = os.path.join(EVAL_DIR, 'low', fname)
            high_p = os.path.join(EVAL_DIR, 'high', fname)
            img_l = Image.open(low_p).convert('RGB')
            img_h = Image.open(high_p).convert('RGB')

            t_l = TF.to_tensor(img_l).unsqueeze(0).to(device)
            h, w = t_l.shape[2], t_l.shape[3]
            mul = 16
            H, W = ((h + mul - 1) // mul) * mul, ((w + mul - 1) // mul) * mul
            pad = F.pad(t_l, (0, W - w, 0, H - h), 'reflect')

            with torch.amp.autocast('cuda'):
                pred = model(pad)
            pred = torch.clamp(pred[:, :, :h, :w], 0, 1)

            np_pred = pred[0].cpu().permute(1, 2, 0).numpy()
            np_gt = (np.array(img_h) / 255.0).astype(np.float32)

            p_val = psnr(np_gt, np_pred, data_range=1.0)
            s_val = ssim(np_gt, np_pred, channel_axis=2, data_range=1.0)

            lab_pred = color.rgb2lab(np_pred)
            lab_gt = color.rgb2lab(np_gt)
            de_map = color.deltaE_ciede2000(lab_gt, lab_pred)
            de_val = float(np.mean(de_map))

            results.append({
                'filename': fname,
                'psnr': float(p_val),
                'ssim': float(s_val),
                'delta_e00': float(de_val),
                'pred_rgb': (np_pred * 255.0).astype(np.uint8),
                'gt_rgb': (np_gt * 255.0).astype(np.uint8),
                'low_rgb': np.array(img_l),
                'de_map': de_map
            })

    mean_psnr = float(np.mean([r['psnr'] for r in results]))
    mean_ssim = float(np.mean([r['ssim'] for r in results]))
    mean_de00 = float(np.mean([r['delta_e00'] for r in results]))

    return {
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'mean_delta_e00': mean_de00,
        'per_image': results
    }

# -------------------------------------------------------------
# 4. Evaluation of Baseline A (Reference Anchor)
# -------------------------------------------------------------
def inspect_baseline_a(device):
    print("\n" + "=" * 65)
    print("   [BƯỚC 1] ĐÁNH GIÁ CHI TIẾT BASELINE A (MỐC THAM CHIẾU)")
    print("=" * 65)

    config_a_dir = os.path.join(ABLATION_ROOT, "config_A_baseline")
    os.makedirs(config_a_dir, exist_ok=True)

    baseline_ckpt = os.path.join(BASE_DIR, "training_50epochs_results", "checkpoints", "model_bestPSNR.pth")
    model = LLFormer(
        inp_channels=3, out_channels=3, dim=16,
        num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', attention=True, skip=False
    ).to(device)

    ckpt = torch.load(baseline_ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"--> Đã nạp Baseline Checkpoint: {baseline_ckpt} (Epoch {ckpt.get('epoch', 45)})")

    eval_res = evaluate_model_on_eval15(model, device)
    print(f"\n--> KẾT QUẢ BASELINE A:")
    print(f"    PSNR trung bình:      {eval_res['mean_psnr']:.2f} dB")
    print(f"    SSIM trung bình:      {eval_res['mean_ssim']:.4f}")
    print(f"    ΔE00 (Sai khác màu): {eval_res['mean_delta_e00']:.2f}")

    print("\n    Bảng chi tiết từng ảnh:")
    print(f"    {'Tên ảnh':8s} | {'PSNR (dB)':9s} | {'SSIM':6s} | {'ΔE00':6s} | {'Đánh giá màu sắc'}")
    print("    " + "-" * 60)
    for r in eval_res['per_image']:
        status = "Bình thường"
        if r['delta_e00'] > 15.0:
            status = "*** LỆCH MÀU NẶNG ***"
        elif r['delta_e00'] > 12.0:
            status = "Lệch màu đáng chú ý"
        elif r['delta_e00'] < 8.0:
            status = "Màu chuẩn, sát GT"
        print(f"    {r['filename']:8s} | {r['psnr']:9.2f} | {r['ssim']:6.4f} | {r['delta_e00']:6.2f} | {status}")

    # Save summary json (without large numpy arrays)
    summary_data = {
        'config': 'A_baseline',
        'loss_type': 'SmoothL1_RGB',
        'mean_psnr': eval_res['mean_psnr'],
        'mean_ssim': eval_res['mean_ssim'],
        'mean_delta_e00': eval_res['mean_delta_e00'],
        'per_image_metrics': [{
            'filename': r['filename'],
            'psnr': r['psnr'],
            'ssim': r['ssim'],
            'delta_e00': r['delta_e00']
        } for r in eval_res['per_image']]
    }
    with open(os.path.join(config_a_dir, "metrics_baseline_a.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)

    return eval_res

# -------------------------------------------------------------
# 5. Training Configuration B: Baseline + HSV Loss
# -------------------------------------------------------------
def train_config_b_hsv(device, total_epochs=50, lambda_hsv=0.15):
    print("\n" + "=" * 65)
    print("   [BƯỚC 2] HUẤN LUYỆN CẤU HÌNH B: BASELINE + HSV LOSS")
    print(f"   Hệ số λ_HSV = {lambda_hsv} (Continuous Circular Chrominance)")
    print("=" * 65)

    config_b_dir = os.path.join(ABLATION_ROOT, "config_B_hsv")
    ckpt_dir = os.path.join(config_b_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Identical seed & setup as baseline
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = LLFormer(
        inp_channels=3, out_channels=3, dim=16,
        num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', attention=True, skip=False
    ).to(device)

    train_dataset = FastRAMDataset(TRAIN_DIR, patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)

    from transform.data_RGB import get_validation_data
    val_dataset = get_validation_data(EVAL_DIR, {'patch_size': 128})
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    criterion_l1 = nn.SmoothL1Loss()

    history = {'epoch': [], 'loss': [], 'loss_l1': [], 'loss_hsv': [], 'val_psnr': [], 'val_ssim': [], 'lr': [], 'time': []}
    best_psnr = -1.0
    start_time_all = time.time()

    print(f"--> Bắt đầu huấn luyện {total_epochs} epochs từ Epoch 0...")

    for epoch in range(total_epochs):
        t0 = time.time()
        model.train()
        running_tot, running_l1, running_hsv = 0.0, 0.0, 0.0

        for target, inp in train_loader:
            for p in model.parameters(): p.grad = None
            target = target.to(device, non_blocking=True)
            inp = inp.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                restored = model(inp)
                l_l1 = criterion_l1(restored, target)
                l_hsv = chroma_hsv_loss(torch.clamp(restored, 0, 1), target)
                loss = l_l1 + lambda_hsv * l_hsv

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_tot += loss.item()
            running_l1 += l_l1.item()
            running_hsv += l_hsv.item()

        n_batches = len(train_loader)
        ep_loss = running_tot / n_batches
        ep_l1 = running_l1 / n_batches
        ep_hsv = running_hsv / n_batches

        # Validation on eval15
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for data_val in val_loader:
                tar = data_val[0].to(device, non_blocking=True)
                inp = data_val[1].to(device, non_blocking=True)
                h, w = tar.shape[2], tar.shape[3]
                with torch.amp.autocast('cuda'):
                    res = model(inp)
                res = torch.clamp(res[:, :, :h, :w].float(), 0, 1)
                psnr_list.append(utils.torchPSNR(res[0], tar[0]).item())
                ssim_list.append(utils.torchSSIM(res, tar).item())

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time = time.time() - t0

        torch.cuda.empty_cache()

        history['epoch'].append(epoch)
        history['loss'].append(ep_loss)
        history['loss_l1'].append(ep_l1)
        history['loss_hsv'].append(ep_hsv)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(cur_lr)
        history['time'].append(ep_time)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim
            }, os.path.join(ckpt_dir, "model_bestPSNR.pth"))

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': val_psnr,
            'ssim': val_ssim
        }, os.path.join(ckpt_dir, "model_latest.pth"))

        with open(os.path.join(config_b_dir, "training_progress.json"), 'w') as f:
            json.dump(history, f, indent=2)

        star = " *** NEW BEST! ***" if is_best else ""
        print(f"--> [Config B Ep {epoch:02d}/49] Loss: {ep_loss:.4f} (L1: {ep_l1:.4f}, HSV: {ep_hsv:.4f}) | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Time: {ep_time:.1f}s{star}")

    total_duration = time.time() - start_time_all
    print(f"\n--> HOÀN THÀNH HUẤN LUYỆN CONFIG B trong {total_duration/60:.1f} phút! Peak PSNR: {best_psnr:.2f} dB")

    # 6. Full Test Evaluation for Config B
    print("\n--> Đang thực hiện đánh giá chi tiết Config B trên test set (eval15)...")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "model_bestPSNR.pth"), map_location=device)
    model.load_state_dict(best_ckpt['state_dict'])
    eval_b = evaluate_model_on_eval15(model, device)

    eval_b_summary = {
        'config': 'B_baseline_plus_hsv',
        'lambda_hsv': lambda_hsv,
        'training_time_min': total_duration / 60.0,
        'mean_psnr': eval_b['mean_psnr'],
        'mean_ssim': eval_b['mean_ssim'],
        'mean_delta_e00': eval_b['mean_delta_e00'],
        'per_image_metrics': [{
            'filename': r['filename'],
            'psnr': r['psnr'],
            'ssim': r['ssim'],
            'delta_e00': r['delta_e00']
        } for r in eval_b['per_image']]
    }
    with open(os.path.join(config_b_dir, "metrics_config_b.json"), 'w') as f:
        json.dump(eval_b_summary, f, indent=2)

    return eval_b, total_duration

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Ablation Pipeline Runner on Device: {device} ({torch.cuda.get_device_name(0)})")

    # Bước 1: Khảo sát Baseline A
    eval_a = inspect_baseline_a(device)

    # Bước 2: Huấn luyện và khảo sát Config B
    eval_b, time_b = train_config_b_hsv(device, total_epochs=50, lambda_hsv=0.15)

    print("\n" + "=" * 65)
    print("   HOÀN TẤT THÍ NGHIỆM ABLATION BƯỚC ĐẦU (A vs B)")
    print("=" * 65)

if __name__ == '__main__':
    main()
