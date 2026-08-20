"""Multi-modal solar PV forecasting model.

Three encoders feeding a fusion + regression head, exactly per the paper:

  LSTMEncoder  : 2-layer LSTM, hidden=128, dropout=0.2 between layers.
                 Output = last-layer final hidden state, vector R^128.
  MetEncoder   : FC(5->128) -> BN -> ReLU -> FC(128->64).  Output R^64.
  CNNEncoder   : 3 conv blocks, filters 32/64/128, each block
                 (3x3 conv pad=1, BN, ReLU, 2x2 avg-pool), then global avg pool.
                 Output R^128.

Two fusion variants:
  ConcatFusion    : [z_ts; z_met; z_img] in R^320  (paper's main model)
  AttentionFusion : projects each encoding to a common 128-dim space, then
                    softmax-weighted sum (paper's attention variant)

Regression head: Dense(d->256) -> ReLU -> Dropout -> Dense(256->128) ->
                 ReLU -> Dropout -> Linear(128->1).

Ablation flags use_ts/use_met/use_img let the train script run paper Table II
(7 modality combinations) by toggling branches.

Init per paper:
  Linear weights : Kaiming uniform
  LSTM weight_hh : orthogonal
  LSTM weight_ih : Kaiming uniform
  LSTM forget-bias = 1, other biases = 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Set forget-gate bias to 1 (helps gradient flow at init)
                hidden = p.shape[0] // 4
                p.data[hidden:2 * hidden].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 24, 6) -> last-layer final hidden (B, 128)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class MetEncoder(nn.Module):
    """FC -> BN -> ReLU -> FC, mapping R^5 to R^64."""

    def __init__(self, in_features: int = 5, hidden: int = 128, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.out_dim = out_features
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5)
        h = F.relu(self.bn(self.fc1(x)))
        return self.fc2(h)


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.AvgPool2d(2)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNNEncoder(nn.Module):
    """3 conv blocks (32, 64, 128 filters) -> Global Average Pool -> R^128."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = _ConvBlock(in_channels, 32)
        self.block2 = _ConvBlock(32, 64)
        self.block3 = _ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 64, 64) -> (B, 32, 32, 32) -> (B, 64, 16, 16) -> (B, 128, 8, 8)
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        return self.gap(h).flatten(1)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
class ConcatFusion(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.out_dim = sum(dims)

    def forward(self, encodings: list[torch.Tensor]):
        return torch.cat(encodings, dim=1), None


class AttentionFusion(nn.Module):
    """Project each encoding to common dim, then softmax-weighted sum.

    Returns the fused vector and the per-sample attention weights so we can
    later analyse which modality the model attends to.
    """

    def __init__(self, dims: list[int], common_dim: int = 128):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, common_dim) for d in dims])
        self.attn = nn.Linear(common_dim * len(dims), len(dims))
        self.out_dim = common_dim
        for proj in self.projections:
            nn.init.kaiming_uniform_(proj.weight, nonlinearity="relu")
            nn.init.zeros_(proj.bias)
        nn.init.kaiming_uniform_(self.attn.weight, nonlinearity="linear")
        nn.init.zeros_(self.attn.bias)

    def forward(self, encodings: list[torch.Tensor]):
        projected = [p(e) for p, e in zip(self.projections, encodings)]
        stacked = torch.stack(projected, dim=1)            # (B, M, D)
        ctx = stacked.flatten(1)                           # (B, M*D)
        alpha = F.softmax(self.attn(ctx), dim=-1)          # (B, M)
        fused = (stacked * alpha.unsqueeze(-1)).sum(dim=1) # (B, D)
        return fused, alpha


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    """Dense(d->256) -> ReLU -> DO -> Dense(256->128) -> ReLU -> DO -> Linear(128->1)."""

    def __init__(self, in_features: int, hidden1: int = 256, hidden2: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.fc1(x)))
        h = self.dropout(F.relu(self.fc2(h)))
        return self.fc3(h).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
class MultiModalSolarModel(nn.Module):
    """Main model with toggleable modalities for ablation runs."""

    def __init__(self, fusion: str = "concat",
                 use_ts: bool = True, use_met: bool = True, use_img: bool = True,
                 dropout_head: float = 0.2):
        super().__init__()
        if not (use_ts or use_met or use_img):
            raise ValueError("at least one modality must be enabled")

        self.use_ts = use_ts
        self.use_met = use_met
        self.use_img = use_img
        self.fusion_type = fusion

        encoders = []
        dims = []
        if use_ts:
            self.lstm_enc = LSTMEncoder()
            encoders.append("ts")
            dims.append(self.lstm_enc.out_dim)
        if use_met:
            self.met_enc = MetEncoder()
            encoders.append("met")
            dims.append(self.met_enc.out_dim)
        if use_img:
            self.cnn_enc = CNNEncoder()
            encoders.append("img")
            dims.append(self.cnn_enc.out_dim)
        self._encoder_order = encoders
        self._encoder_dims = dims

        # Fusion is degenerate with one modality -> use whatever single vector we have
        if len(encoders) == 1:
            self.fusion = ConcatFusion(dims)  # acts as identity on one input
        elif fusion == "concat":
            self.fusion = ConcatFusion(dims)
        elif fusion == "attention":
            self.fusion = AttentionFusion(dims)
        else:
            raise ValueError(f"unknown fusion type: {fusion}")

        self.head = RegressionHead(in_features=self.fusion.out_dim, dropout=dropout_head)

    def encode(self, x_ts=None, x_met=None, x_img=None):
        encodings = []
        if self.use_ts:
            encodings.append(self.lstm_enc(x_ts))
        if self.use_met:
            encodings.append(self.met_enc(x_met))
        if self.use_img:
            encodings.append(self.cnn_enc(x_img))
        return encodings

    def forward(self, x_ts=None, x_met=None, x_img=None, return_alpha: bool = False):
        encodings = self.encode(x_ts, x_met, x_img)
        fused, alpha = self.fusion(encodings)
        y = self.head(fused)
        if return_alpha:
            return y, alpha
        return y

    @property
    def encoder_order(self):
        return list(self._encoder_order)


# ---------------------------------------------------------------------------
# Smoke test entrypoint
# ---------------------------------------------------------------------------
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/pune_500kw_hourly.csv")
    ap.add_argument("--npz", default="data/satellite_patches.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("model")

    from dataset import build_datasets
    from torch.utils.data import DataLoader

    log.info("Loading one batch from the train split for a forward-pass smoke test")
    ds_train, _, _, stats, _ = build_datasets(args.csv, args.npz)
    loader = DataLoader(ds_train, batch_size=8, shuffle=True)
    x_ts, x_met, x_img, y_norm, y_kw = next(iter(loader))
    x_ts, x_met, x_img = x_ts.to(args.device), x_met.to(args.device), x_img.to(args.device)
    log.info("Batch shapes  x_ts=%s x_met=%s x_img=%s", tuple(x_ts.shape),
             tuple(x_met.shape), tuple(x_img.shape))

    log.info("---- Concat-fusion (paper main model) ----")
    model_concat = MultiModalSolarModel(fusion="concat").to(args.device)
    log.info("Trainable params: %s", f"{_count_params(model_concat):,}")
    y_pred = model_concat(x_ts, x_met, x_img)
    log.info("Output shape=%s  mean=%.3f std=%.3f finite=%s",
             tuple(y_pred.shape), float(y_pred.mean()),
             float(y_pred.std()), bool(torch.isfinite(y_pred).all()))

    log.info("---- Attention-fusion variant ----")
    model_attn = MultiModalSolarModel(fusion="attention").to(args.device)
    log.info("Trainable params: %s", f"{_count_params(model_attn):,}")
    y_pred, alpha = model_attn(x_ts, x_met, x_img, return_alpha=True)
    log.info("Output shape=%s   alpha shape=%s", tuple(y_pred.shape), tuple(alpha.shape))
    log.info("Mean alpha (ts, met, img) at init: %s",
             [round(float(v), 3) for v in alpha.mean(dim=0).tolist()])

    log.info("---- Backward pass + grad check (concat) ----")
    target = y_norm.to(args.device)
    y_pred = model_concat(x_ts, x_met, x_img)
    loss = F.mse_loss(y_pred, target)
    loss.backward()
    n_with_grad = sum(int(p.grad is not None) for p in model_concat.parameters())
    n_total = sum(1 for _ in model_concat.parameters())
    log.info("Loss=%.4f  parameters with grad: %d/%d", float(loss), n_with_grad, n_total)

    log.info("---- Ablation reachability ----")
    for cfg in [
        dict(use_ts=True, use_met=False, use_img=False),
        dict(use_ts=False, use_met=True, use_img=False),
        dict(use_ts=False, use_met=False, use_img=True),
        dict(use_ts=True, use_met=True, use_img=False),
        dict(use_ts=True, use_met=False, use_img=True),
        dict(use_ts=False, use_met=True, use_img=True),
        dict(use_ts=True, use_met=True, use_img=True),
    ]:
        m = MultiModalSolarModel(fusion="concat", **cfg).to(args.device)
        out = m(
            x_ts=x_ts if cfg["use_ts"] else None,
            x_met=x_met if cfg["use_met"] else None,
            x_img=x_img if cfg["use_img"] else None,
        )
        log.info("  %s -> output %s  params=%s",
                 "+".join(m.encoder_order), tuple(out.shape),
                 f"{_count_params(m):,}")


if __name__ == "__main__":
    main()
