#!/usr/bin/env python3
"""
Train a VAE that compresses Stage-2 features to a 16-dim latent space.
Optimized to use CUDA (RTX A2000) when available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    latent_dim: int = 16
    hidden_dims: Tuple[int, int, int] = (512, 256, 128)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 120
    patience: int = 25
    beta_target: float = 2.0
    warmup_epochs: int = 40
    free_bits: float = 2.0
    grad_clip: float = 1.0
    seed: int = 42
    val_size: float = 0.1
    use_amp: bool = True


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int, int, int]) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(h3, latent_dim)
        self.logvar_head = nn.Linear(h3, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.ReLU(),
            nn.Linear(h3, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.mean(kl_per_dim, dim=0)
    kl = torch.sum(torch.clamp(kl_per_dim, min=free_bits / mu.shape[1]))
    loss = recon + beta * kl
    return loss, recon, kl


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(x.astype(np.float32))
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    free_bits: float,
    use_amp: bool,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    batches = 0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                x_hat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=beta, free_bits=free_bits)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu())
        total_recon += float(recon.detach().cpu())
        total_kl += float(kl.detach().cpu())
        batches += 1

    return {
        "loss": total_loss / batches,
        "recon": total_recon / batches,
        "kl": total_kl / batches,
    }


@torch.no_grad()
def encode_all(model: VAE, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = make_loader(x, batch_size=batch_size, shuffle=False)
    all_z: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        mu, _ = model.encode(xb)
        all_z.append(mu.detach().cpu().numpy())
    return np.vstack(all_z)


def read_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    x_train_path = data_dir / "stage2_X_train.parquet"
    x_test_path = data_dir / "stage2_X_test.parquet"
    if not x_train_path.exists() or not x_test_path.exists():
        raise FileNotFoundError(f"Expected stage2 parquet files in: {data_dir}")

    x_train = pd.read_parquet(x_train_path).values.astype(np.float32)
    x_test = pd.read_parquet(x_test_path).values.astype(np.float32)
    y_train_path = data_dir / "stage2_y_train.parquet"
    y_test_path = data_dir / "stage2_y_test.parquet"
    y_train = pd.read_parquet(y_train_path).values.reshape(-1) if y_train_path.exists() else None
    y_test = pd.read_parquet(y_test_path).values.reshape(-1) if y_test_path.exists() else None
    return x_train, x_test, y_train, y_test


def save_latents(z: np.ndarray, out_path: Path) -> None:
    cols = [f"z{i}" for i in range(z.shape[1])]
    pd.DataFrame(z, columns=cols).to_parquet(out_path, index=False)


def save_tsne_plot(z: np.ndarray, y: np.ndarray | None, out_path: Path, seed: int) -> None:
    max_points = 12000
    if z.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(z.shape[0], size=max_points, replace=False)
        z_plot = z[idx]
        y_plot = y[idx] if y is not None else None
    else:
        z_plot = z
        y_plot = y

    perplexity = min(30, max(5, (z_plot.shape[0] // 100) - 1))
    z_2d = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(z_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    if y_plot is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_plot, s=5, alpha=0.6, cmap="tab10")
        fig.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6)
    ax.set_title("VAE v3 Latent t-SNE (16-dim -> 2D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_angle_distribution_plot(z: np.ndarray, out_path: Path) -> None:
    # Map latent values to rotation angles in [-pi, pi]
    angles = np.pi * np.tanh(z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(angles.reshape(-1), bins=80, color="#1f77b4", alpha=0.85)
    axes[0].set_title("All Angle Values Distribution")
    axes[0].set_xlabel("Angle (radians)")
    axes[0].set_ylabel("Count")

    dim_stds = np.std(angles, axis=0)
    axes[1].bar(np.arange(angles.shape[1]), dim_stds, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Per-Latent Angle Std Dev")
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Std Dev")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE v3 with 16-dim latent compression.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("PreProcessing") / "stage_2_with_zero_v2",
        help="Directory containing stage2_X_train.parquet and stage2_X_test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("VAE") / "vae_v3_output_16",
        help="Output directory",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension (use 16 for 16-bit quantum stage)")
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir, output_dir=args.output_dir, latent_dim=args.latent_dim)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, _ = read_data(cfg.data_dir)
    x_train_fit, x_val_fit = train_test_split(x_train, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True)

    train_loader = make_loader(x_train_fit, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val_fit, cfg.batch_size, shuffle=False)

    model = VAE(input_dim=x_train.shape[1], latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
    )

    history: Dict[str, List[float]] = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": [], "val_recon": [], "val_kl": []}
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_path = cfg.output_dir / "vae_v3_16_best.pt"

    for epoch in range(cfg.max_epochs):
        beta = min(cfg.beta_target, cfg.beta_target * (epoch + 1) / max(1, cfg.warmup_epochs))
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            free_bits=cfg.free_bits,
            use_amp=cfg.use_amp,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            free_bits=cfg.free_bits,
            use_amp=cfg.use_amp,
            train=False,
        )

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    z_train = encode_all(model, x_train, device=device, batch_size=cfg.batch_size)
    z_test = encode_all(model, x_test, device=device, batch_size=cfg.batch_size)

    save_latents(z_train, cfg.output_dir / "vae_v3_z_train.parquet")
    save_latents(z_test, cfg.output_dir / "vae_v3_z_test.parquet")
    save_tsne_plot(z_train, y_train, cfg.output_dir / "vae-v3_tsne.png", seed=cfg.seed)
    save_angle_distribution_plot(z_train, cfg.output_dir / "vae-v3_angle_distributions.png")

    latent_stats = {
        "z_train_shape": list(z_train.shape),
        "z_test_shape": list(z_test.shape),
        "z_train_mean": z_train.mean(axis=0).tolist(),
        "z_train_std": z_train.std(axis=0).tolist(),
    }
    with open(cfg.output_dir / "vae_v3_16_latent_stats.json", "w", encoding="utf-8") as f:
        json.dump(latent_stats, f, indent=2)

    cfg_dict = asdict(cfg)
    cfg_dict["data_dir"] = str(cfg.data_dir)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    cfg_dict["hidden_dims"] = list(cfg.hidden_dims)
    cfg_dict["device"] = str(device)
    cfg_dict["input_dim"] = int(x_train.shape[1])
    cfg_dict["best_epoch"] = int(best_epoch)
    cfg_dict["best_val_loss"] = float(best_val)
    with open(cfg.output_dir / "vae_v3_16_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    with open(cfg.output_dir / "vae_v3_16_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[OK] Trained VAE v3 with latent_dim={cfg.latent_dim} on device={device}")
    print(f"[OK] Saved latents: {cfg.output_dir / 'vae_v3_z_train.parquet'}")
    print(f"[OK] Saved latents: {cfg.output_dir / 'vae_v3_z_test.parquet'}")
    print(f"[OK] Saved plot: {cfg.output_dir / 'vae-v3_tsne.png'}")
    print(f"[OK] Saved plot: {cfg.output_dir / 'vae-v3_angle_distributions.png'}")


if __name__ == "__main__":
    main()
