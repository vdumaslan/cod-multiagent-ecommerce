from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FTTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_token: int = 32, n_heads: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.feature_weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # Numerical feature tokenization (FT-Transformer style).
        tokens = x_num.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        h = self.encoder(x)
        out = self.head(h[:, 0, :]).squeeze(-1)
        return out


def _to_tensor(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = df[feature_cols].astype(float).to_numpy(dtype=np.float32)
    y = df[target_col].astype(float).to_numpy(dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def _scale(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stats: dict[str, dict[str, float]] = {}
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in feature_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std()) if float(train_df[col].std()) > 1e-6 else 1.0
        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return train_df, test_df, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", default="seller-copilot/artifacts/data/pricing_train.parquet")
    parser.add_argument("--test-parquet", default="seller-copilot/artifacts/data/pricing_test.parquet")
    parser.add_argument("--target-col", default="target_price")
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["price", "avg_rating", "review_count", "positive_ratio", "rating_price_ratio"],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/pricing")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    required = set(args.feature_cols + [args.target_col])
    if not required.issubset(set(train_df.columns)):
        raise RuntimeError(f"Missing required columns in train parquet: {sorted(required - set(train_df.columns))}")
    if not required.issubset(set(test_df.columns)):
        raise RuntimeError(f"Missing required columns in test parquet: {sorted(required - set(test_df.columns))}")

    train_df = train_df.dropna(subset=list(required)).copy()
    test_df = test_df.dropna(subset=list(required)).copy()
    if len(train_df) < 500 or len(test_df) < 100:
        raise RuntimeError("Insufficient rows for FT-Transformer training.")

    train_df, test_df, scaler = _scale(train_df, test_df, args.feature_cols)
    x_train, y_train = _to_tensor(train_df, args.feature_cols, args.target_col)
    x_test, y_test = _to_tensor(test_df, args.feature_cols, args.target_col)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    model = FTTransformerRegressor(
        n_features=len(args.feature_cols),
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    epoch_losses: list[float] = []
    for _ in range(args.epochs):
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        epoch_losses.append(total / max(1, count))

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test.to(device)).cpu().numpy()
    y_true = y_test.cpu().numpy()

    mae = float(mean_absolute_error(y_true, pred_test))
    rmse = float(mean_squared_error(y_true, pred_test, squared=False))
    r2 = float(r2_score(y_true, pred_test))
    mape = float(np.mean(np.abs((y_true - pred_test) / np.clip(y_true, 1e-3, None))) * 100.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": args.feature_cols,
            "target_col": args.target_col,
            "scaler": scaler,
            "config": {
                "d_token": args.d_token,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
            },
        },
        out_dir / "ft_transformer.pt",
    )

    payload = {
        "model_id": "FT-Transformer",
        "status": "trained",
        "device": device,
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape},
        "loss_curve": epoch_losses,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
