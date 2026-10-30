"""
Resume training Config B (Baseline + HSV Loss) from last checkpoint.
"""
import os, sys, time, json, random, numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import skimage.color as color

from model.LLFormer import LLFormer
from transform.data_RGB import get_validation_data
import utils

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
torch.backends.cudnn.benchmark = True

EVAL_DIR      = r"c:\Users\Admin\Documents\LLFormer\LOLdataset\eval15"
TRAIN_DIR     = r"c:\Users\Admin\Documents\LLFormer\LOLdataset\our485"
CONFIG_B_DIR  = r"c:\Users\Admin\Documents\LLFormer\ablation_results\config_B_hsv"
CKPT_DIR      = os.path.join(CONFIG_B_DIR, "checkpoints")
ARTIFACTS_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\ebffbe2e-1ff6-4ae2-9aba-4a5154854224"
PROGRESS_JSON = os.path.join(CONFIG_B_DIR, "training_progress.json")
METRICS_JSON  = os.path.join(CONFIG_B_DIR, "metrics_config_b.json")

TOTAL_EPOCHS = 50
LAMBDA_HSV   = 0.15

# ── Losses ──────────────────────────────────────────────────
def rgb_to_hsv_diff(rgb, eps=1e-7):
    r, g, b = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
    cmax, _ = torch.max(rgb, dim=1, keepdim=True)
    cmin, _ = torch.min(rgb, dim=1, keepdim=True)
    delta = cmax - cmin + eps
    r_max = (cmax == r).float()
    g_max = ((cmax == g) & (cmax != r)).float()
    b_max = ((cmax != r) & (cmax != g)).float()
    dh = (g - b) / delta
    dh = dh - 6.0 * torch.floor(dh / 6.0)
    h = r_max * dh + g_max * (((b - r) / delta) + 2.0) + b_max * (((r - g) / delta) + 4.0)
    h /= 6.0
    s = (cmax - cmin) / (cmax + eps)
    return torch.cat([h, s, cmax], dim=1)

def chroma_hsv_loss(pred, gt):
    hp = rgb_to_hsv_diff(pred); hg = rgb_to_hsv_diff(gt)
    s_p, h_p = hp[:,1:2], hp[:,0:1]
    s_g, h_g = hg[:,1:2], hg[:,0:1]
    PI2 = 2 * 3.1415926535
    cx_p = s_p * torch.cos(PI2 * h_p); cy_p = s_p * torch.sin(PI2 * h_p)
    cx_g = s_g * torch.cos(PI2 * h_g); cy_g = s_g * torch.sin(PI2 * h_g)
    return F.l1_loss(cx_p, cx_g) + F.l1_loss(cy_p, cy_g)

# ── Dataset ──────────────────────────────────────────────────
class FastRAMDataset(Dataset):
    def __init__(self, rgb_dir, patch_size=128):
        self.ps = patch_size
        low_dir  = os.path.join(rgb_dir, 'low')
        high_dir = os.path.join(rgb_dir, 'high')
        inp  = sorted([f for f in os.listdir(low_dir)  if f.lower().endswith(('.png','.jpg'))])
        tar  = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('.png','.jpg'))])
        print(f"--> Loading {len(inp)} image pairs into RAM...")
        t0 = time.time()
        self.samples = [(TF.to_tensor(Image.open(os.path.join(high_dir, fh)).convert('RGB')),
                         TF.to_tensor(Image.open(os.path.join(low_dir,  fl)).convert('RGB')))
                        for fl, fh in zip(inp, tar)]
        print(f"--> Loaded in {time.time()-t0:.1f}s")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        tar, inp = self.samples[idx]
        ps = self.ps; hh, ww = tar.shape[1], tar.shape[2]
        rr = random.randint(0, hh - ps); cc = random.randint(0, ww - ps)
        ct = tar[:, rr:rr+ps, cc:cc+ps]; ci = inp[:, rr:rr+ps, cc:cc+ps]
        aug = random.randint(0, 4)
        if aug == 1: ct, ci = ct.flip(1), ci.flip(1)
        elif aug == 2: ct, ci = ct.flip(2), ci.flip(2)
        elif aug == 3: ct, ci = torch.rot90(ct,1,[1,2]), torch.rot90(ci,1,[1,2])
        return ct, ci

# ── Full Evaluation ──────────────────────────────────────────
def evaluate_full(model, device):
    model.eval()
    low_files = sorted([f for f in os.listdir(os.path.join(EVAL_DIR,'low')) if f.endswith(('.png','.jpg'))])
    results = []
    with torch.no_grad():
        for fname in low_files:
            img_l = Image.open(os.path.join(EVAL_DIR,'low', fname)).convert('RGB')
            img_h = Image.open(os.path.join(EVAL_DIR,'high',fname)).convert('RGB')
            t_l = TF.to_tensor(img_l).unsqueeze(0).to(device)
            h, w = t_l.shape[2], t_l.shape[3]
            mul = 16
            H, W = ((h+mul-1)//mul)*mul, ((w+mul-1)//mul)*mul
            pad = F.pad(t_l, (0, W-w, 0, H-h), 'reflect')
            with torch.amp.autocast('cuda'):
                pred = model(pad)
            pred = torch.clamp(pred[:,:,:h,:w], 0, 1)
            np_pred = pred[0].cpu().permute(1,2,0).numpy()
            np_gt   = (np.array(img_h)/255.0).astype(np.float32)
            p  = psnr_fn(np_gt, np_pred, data_range=1.0)
            s  = ssim_fn(np_gt, np_pred, channel_axis=2, data_range=1.0)
            de = float(np.mean(color.deltaE_ciede2000(color.rgb2lab(np_gt), color.rgb2lab(np_pred))))
            results.append({'filename': fname, 'psnr': float(p), 'ssim': float(s), 'delta_e00': float(de)})
    return results

# ── Main ─────────────────────────────────────────────────────
def main():
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*65)
    print("   Config B Resume: Baseline + HSV Chroma Loss")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0)})")
    print("="*65)

    # Load history
    with open(PROGRESS_JSON) as f:
        history = json.load(f)
    start_epoch = history['epoch'][-1] + 1
    best_psnr   = max(history['val_psnr'])
    print(f"--> Resuming from epoch {start_epoch} (best PSNR so far: {best_psnr:.2f} dB)")

    # Model
    model = LLFormer(
        inp_channels=3, out_channels=3, dim=16,
        num_blocks=[2,4,8,16], num_refinement_blocks=2,
        heads=[1,2,4,8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', attention=True, skip=False
    ).to(device)

    latest_ckpt = os.path.join(CKPT_DIR, 'model_latest.pth')
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print(f"--> Loaded checkpoint: epoch {ckpt['epoch']}, PSNR {ckpt['psnr']:.2f} dB")

    # Data
    train_dataset = FastRAMDataset(TRAIN_DIR, patch_size=128)
    train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_dataset   = get_validation_data(EVAL_DIR, {'patch_size': 128})
    val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Optimizer — restore LR position via last_epoch
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9,0.999), eps=1e-8)
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
    except: pass
    criterion = nn.SmoothL1Loss()
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6, last_epoch=start_epoch-1)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        t0 = time.time()
        model.train()
        run_tot = run_l1 = run_hsv = 0.0

        for target, inp in train_loader:
            for p in model.parameters(): p.grad = None
            target = target.to(device, non_blocking=True)
            inp    = inp.to(device,    non_blocking=True)
            with torch.amp.autocast('cuda'):
                restored = model(inp)
                l_l1  = criterion(restored, target)
                l_hsv = chroma_hsv_loss(torch.clamp(restored, 0, 1), target)
                loss  = l_l1 + LAMBDA_HSV * l_hsv
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_tot += loss.item(); run_l1 += l_l1.item(); run_hsv += l_hsv.item()

        nb = len(train_loader)
        ep_loss = run_tot/nb; ep_l1 = run_l1/nb; ep_hsv = run_hsv/nb

        # Validation
        model.eval()
        plist, slist = [], []
        with torch.no_grad():
            for dv in val_loader:
                tar = dv[0].to(device, non_blocking=True)
                inp = dv[1].to(device, non_blocking=True)
                h, w = tar.shape[2], tar.shape[3]
                with torch.amp.autocast('cuda'):
                    res = model(inp)
                res = torch.clamp(res[:,:,:h,:w].float(), 0, 1)
                plist.append(utils.torchPSNR(res[0], tar[0]).item())
                slist.append(utils.torchSSIM(res, tar).item())

        val_psnr = float(np.mean(plist))
        val_ssim = float(np.mean(slist))
        cur_lr   = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_time  = time.time() - t0
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
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'psnr': val_psnr, 'ssim': val_ssim},
                       os.path.join(CKPT_DIR, 'model_bestPSNR.pth'))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr': val_psnr, 'ssim': val_ssim},
                   os.path.join(CKPT_DIR, 'model_latest.pth'))
        with open(PROGRESS_JSON, 'w') as f:
            json.dump(history, f, indent=2)

        star = " *** NEW BEST! ***" if is_best else ""
        print(f"--> [B Ep {epoch:02d}/49] Loss:{ep_loss:.4f} (L1:{ep_l1:.4f} HSV:{ep_hsv:.4f}) "
              f"| PSNR:{val_psnr:.2f}dB | SSIM:{val_ssim:.4f} | {ep_time:.0f}s{star}")

    # ── Final full evaluation with DeltaE00 ──────────────────
    print("\n--> Final full evaluation on eval15 (PSNR + SSIM + DeltaE00)...")
    best_ckpt = torch.load(os.path.join(CKPT_DIR,'model_bestPSNR.pth'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['state_dict'])
    results = evaluate_full(model, device)

    mean_psnr = float(np.mean([r['psnr'] for r in results]))
    mean_ssim = float(np.mean([r['ssim'] for r in results]))
    mean_de00 = float(np.mean([r['delta_e00'] for r in results]))

    summary = {
        'config': 'B_baseline_plus_hsv', 'lambda_hsv': LAMBDA_HSV,
        'best_epoch': int(best_ckpt['epoch']),
        'mean_psnr': mean_psnr, 'mean_ssim': mean_ssim, 'mean_delta_e00': mean_de00,
        'per_image_metrics': results
    }
    with open(METRICS_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*65)
    print("   CONFIG B COMPLETE")
    print(f"   Best Epoch : {best_ckpt['epoch']}")
    print(f"   Mean PSNR  : {mean_psnr:.2f} dB")
    print(f"   Mean SSIM  : {mean_ssim:.4f}")
    print(f"   Mean DeltaE00: {mean_de00:.2f}")
    print("="*65)

    # Auto-generate comparison tables
    print("\n--> Generating ablation comparison tables and figures...")
    import subprocess
    subprocess.run(['python', 'generate_ablation_tables.py'], check=False)

if __name__ == '__main__':
    main()
