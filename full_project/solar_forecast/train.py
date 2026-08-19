"""Training loop for the multi-modal solar forecasting model.

Implements the recipe from the paper:
  - Loss: MSE on normalised target
  - Optimiser: Adam, lr=1e-3, weight_decay=1e-4
  - Schedule: linear warmup 5 epochs (lr/10 -> lr), then cosine annealing to lr/100
  - Gradient clip: norm 1.0
  - Early stopping: val RMSE in kW, patience=12
  - Batch size 64, seed=42
  - CNN blocks 1-2 frozen for first 10 epochs (no ImageNet pre-init - flagged in README)
  - Save best-val checkpoint + per-epoch history CSV

Run:
  python train.py                       # full early-fusion (paper main)
  python train.py --fusion attention    # attention-fusion variant
  python train.py --no-img              # ablation: TS+Met only
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import MultiModalSolarModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_at_epoch(epoch: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup from base_lr/10 -> base_lr, then cosine to base_lr/100."""
    if epoch < warmup:
        frac = (epoch + 1) / max(warmup, 1)
        return base_lr * (0.1 + 0.9 * frac)
    progress = (epoch - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (0.01 + 0.99 * cosine)


def freeze(module: torch.nn.Module, frozen: bool):
    for p in module.parameters():
        p.requires_grad = not frozen


def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    n_loss, sum_loss = 0, 0.0
    preds_kw, targets_kw = [], []
    with torch.no_grad():
        for batch in loader:
            x_ts, x_met, x_img, y_norm, y_kw = [b.to(device) for b in batch]
            pred = model(
                x_ts=x_ts if model.use_ts else None,
                x_met=x_met if model.use_met else None,
                x_img=x_img if model.use_img else None,
            )
            loss = F.mse_loss(pred, y_norm)
            sum_loss += loss.item() * y_norm.size(0)
            n_loss += y_norm.size(0)
            pred_kw = pred * target_std + target_mean
            preds_kw.append(pred_kw.cpu().numpy())
            targets_kw.append(y_kw.cpu().numpy())
    preds_kw = np.concatenate(preds_kw)
    targets_kw = np.concatenate(targets_kw)
    rmse = float(np.sqrt(np.mean((preds_kw - targets_kw) ** 2)))
    mae = float(np.mean(np.abs(preds_kw - targets_kw)))
    ss_res = np.sum((preds_kw - targets_kw) ** 2)
    ss_tot = np.sum((targets_kw - targets_kw.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))
    avg_loss = sum_loss / max(n_loss, 1)
    return avg_loss, rmse, mae, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    ap.add_argument("--npz", default="data/satellite_patches.npz")
    ap.add_argument("--ckpt", default="data/checkpoints/best.pt")
    ap.add_argument("--history", default="data/checkpoints/history.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion", choices=["concat", "attention"], default="concat")
    ap.add_argument("--no-ts", action="store_true")
    ap.add_argument("--no-met", action="store_true")
    ap.add_argument("--no-img", action="store_true")
    ap.add_argument("--cnn-freeze-epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("train")

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)

    log.info("Setting seed=%d", args.seed)
    set_seed(args.seed)
    device = torch.device(args.device)
    log.info("Device: %s", device)

    log.info("Building datasets")
    ds_train, ds_val, ds_test, stats, _ = build_datasets(args.csv, args.npz)
    log.info("Sizes train=%d val=%d test=%d", len(ds_train), len(ds_val), len(ds_test))

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    use_ts = not args.no_ts
    use_met = not args.no_met
    use_img = not args.no_img
    log.info("Modalities  ts=%s met=%s img=%s  fusion=%s",
             use_ts, use_met, use_img, args.fusion)

    model = MultiModalSolarModel(fusion=args.fusion, use_ts=use_ts,
                                 use_met=use_met, use_img=use_img).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %s", f"{n_params:,}")

    # Freeze CNN blocks 1-2 for the warm-up window
    if use_img and args.cnn_freeze_epochs > 0:
        log.info("Freezing CNN blocks 1-2 for first %d epochs", args.cnn_freeze_epochs)
        freeze(model.cnn_enc.block1, True)
        freeze(model.cnn_enc.block2, True)

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    patience = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        # CNN unfreeze handover
        if use_img and args.cnn_freeze_epochs > 0 and epoch == args.cnn_freeze_epochs:
            log.info("Unfreezing CNN blocks 1-2 at epoch %d", epoch)
            freeze(model.cnn_enc.block1, False)
            freeze(model.cnn_enc.block2, False)
            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay,
            )

        # LR schedule
        lr = lr_at_epoch(epoch, args.lr, args.warmup_epochs, args.epochs)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Train pass
        model.train()
        running, n_running = 0.0, 0
        for batch in train_loader:
            x_ts, x_met, x_img, y_norm, _ = [b.to(device) for b in batch]
            pred = model(
                x_ts=x_ts if use_ts else None,
                x_met=x_met if use_met else None,
                x_img=x_img if use_img else None,
            )
            loss = F.mse_loss(pred, y_norm)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            running += loss.item() * y_norm.size(0)
            n_running += y_norm.size(0)
        train_loss = running / max(n_running, 1)

        # Validate
        val_loss, val_rmse, val_mae, val_r2 = evaluate(
            model, val_loader, device, stats.target_mean, stats.target_std)

        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_rmse_kw": val_rmse, "val_mae_kw": val_mae, "val_r2": val_r2,
        })
        elapsed = time.time() - t0
        log.info("epoch %3d | lr=%.5f | train=%.4f val=%.4f "
                 "| val_RMSE=%.2f kW val_MAE=%.2f kW val_R2=%.3f | %.1fs",
                 epoch, lr, train_loss, val_loss, val_rmse, val_mae, val_r2, elapsed)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_rmse_kw": val_rmse,
                "args": vars(args),
                "encoder_order": model.encoder_order,
                "target_mean": stats.target_mean,
                "target_std": stats.target_std,
            }, args.ckpt)
            log.info("  ** new best val RMSE = %.2f kW (saved %s)", val_rmse, args.ckpt)
        else:
            patience += 1
            if patience >= args.patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch, patience)
                break

    pd.DataFrame(history).to_csv(args.history, index=False)
    log.info("History -> %s", args.history)

    # Final test pass with best checkpoint
    log.info("Loading best checkpoint (epoch %d, val_RMSE=%.2f kW)",
             best_epoch, best_val_rmse)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_loss, test_rmse, test_mae, test_r2 = evaluate(
        model, test_loader, device, stats.target_mean, stats.target_std)
    log.info("=" * 70)
    log.info("FINAL TEST RESULTS  (best checkpoint, epoch %d)", best_epoch)
    log.info("=" * 70)
    log.info("  RMSE   = %6.2f kW   (paper proposed Early-Fusion: 6.14)", test_rmse)
    log.info("  MAE    = %6.2f kW   (paper proposed Early-Fusion: 4.87)", test_mae)
    log.info("  R^2    = %6.3f      (paper proposed Early-Fusion: 0.963)", test_r2)
    log.info("  nRMSE  = %6.2f %% of 500 kW nameplate", 100.0 * test_rmse / 500.0)
    log.info("  Train+val time: %.0f sec", time.time() - t0)

    metrics_path = Path(args.ckpt).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val_rmse_kw": best_val_rmse,
        "test_rmse_kw": test_rmse,
        "test_mae_kw": test_mae,
        "test_r2": test_r2,
        "test_nrmse_pct": 100.0 * test_rmse / 500.0,
        "fusion": args.fusion,
        "use_ts": use_ts, "use_met": use_met, "use_img": use_img,
        "params": n_params,
        "epochs_run": len(history),
    }, indent=2))
    log.info("Metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
