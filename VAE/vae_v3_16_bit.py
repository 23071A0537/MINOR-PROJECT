#!/usr/bin/env python3
"""
Train a VAE that compresses Stage-2 features to a 16-dim latent space.
Optimized to use CUDA (RTX A2000) when available.
"""

from __future__ import annotations

import argparse
import json
import time
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


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
    aux_weight: float = 0.7
    n_classes: int = 5
    weighted_sampling: bool = True
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


def clear_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_quantum_angles(mu: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mu) * torch.pi


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits: float,
    y: torch.Tensor | None = None,
    aux_clf: nn.Module | None = None,
    class_weights_tensor: torch.Tensor | None = None,
    aux_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.mean(kl_per_dim, dim=0)
    kl = torch.sum(torch.clamp(kl_per_dim, min=free_bits / mu.shape[1]))
    aux = torch.tensor(0.0, device=x.device)
    if y is not None and aux_clf is not None:
        logits = aux_clf(mu)
        if class_weights_tensor is not None:
            aux = F.cross_entropy(logits, y, weight=class_weights_tensor.to(mu.device))
        else:
            aux = F.cross_entropy(logits, y)
    loss = recon + beta * kl + aux_weight * aux
    return loss, recon, kl, aux


def make_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    y: np.ndarray | None = None,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    x_tensor = torch.from_numpy(x)
    if y is not None:
        y_tensor = torch.from_numpy(y.astype(np.int64, copy=False))
        ds = TensorDataset(x_tensor, y_tensor)
    else:
        ds = TensorDataset(x_tensor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,
    )


def run_epoch(
    model: VAE,
    aux_clf: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    free_bits: float,
    use_amp: bool,
    train: bool,
    class_weights_tensor: torch.Tensor | None = None,
    aux_weight: float = 0.0,
) -> Dict[str, float]:
    if train:
        model.train()
        if aux_clf is not None:
            aux_clf.train()
    else:
        model.eval()
        if aux_clf is not None:
            aux_clf.eval()

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_aux = 0.0
    batches = 0

    for batch in loader:
        has_y = len(batch) == 2
        x = batch[0]
        y = batch[1] if has_y else None
        x = x.to(device, non_blocking=True)
        if y is not None:
            y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                x_hat, mu, logvar = model(x)
                loss, recon, kl, aux = vae_loss(
                    x_hat,
                    x,
                    mu,
                    logvar,
                    beta=beta,
                    free_bits=free_bits,
                    y=y if train else None,
                    aux_clf=aux_clf if train else None,
                    class_weights_tensor=class_weights_tensor,
                    aux_weight=aux_weight if train else 0.0,
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                params = list(model.parameters())
                if aux_clf is not None:
                    params += list(aux_clf.parameters())
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu())
        total_recon += float(recon.detach().cpu())
        total_kl += float(kl.detach().cpu())
        total_aux += float(aux.detach().cpu())
        batches += 1

    return {
        "loss": total_loss / batches,
        "recon": total_recon / batches,
        "kl": total_kl / batches,
        "aux": total_aux / batches,
    }


@torch.no_grad()
def encode_all(model: VAE, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = make_loader(x, batch_size=batch_size, shuffle=False)
    all_z: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        mu, _ = model.encode(xb)
        angles = to_quantum_angles(mu)
        all_z.append(angles.detach().cpu().numpy())
    return np.vstack(all_z)


def read_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    x_train_path = data_dir / "stage2_X_train.parquet"
    x_test_path = data_dir / "stage2_X_test.parquet"
    if not x_train_path.exists() or not x_test_path.exists():
        raise FileNotFoundError(f"Expected stage2 parquet files in: {data_dir}")

    x_train = pd.read_parquet(x_train_path).to_numpy(dtype=np.float32, copy=False)
    x_test = pd.read_parquet(x_test_path).to_numpy(dtype=np.float32, copy=False)
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
    # z is already mapped to quantum angles in [0, pi]
    angles = z

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


def save_training_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="train_recon")
    axes[1].plot(epochs, history["val_recon"], label="val_recon")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recon")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def angle_diversity_pass(z: np.ndarray) -> bool:
    return bool(np.std(z) > 0.05)


def min_class_centroid_distance(z: np.ndarray, y: np.ndarray | None) -> float:
    if y is None:
        return float("nan")
    classes = np.unique(y)
    centroids = {int(c): z[y == c].mean(axis=0) for c in classes}
    min_d = float("inf")
    for i, ci in enumerate(classes):
        for cj in classes[i + 1 :]:
            d = float(np.linalg.norm(centroids[int(ci)] - centroids[int(cj)]))
            min_d = min(min_d, d)
    return min_d


def build_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    classes, counts = np.unique(y, return_counts=True)
    class_freq = {int(c): int(n) for c, n in zip(classes, counts)}
    class_weights = {c: 1.0 / n for c, n in class_freq.items()}
    sample_weights = np.array([class_weights[int(c)] for c in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights_tensor(y: np.ndarray, n_classes: int) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    weights = np.ones(n_classes, dtype=np.float32)
    for c, n in zip(classes, counts):
        weights[int(c)] = total / (len(classes) * float(n))
    return torch.tensor(weights, dtype=torch.float32)


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
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--aux-weight", type=float, default=0.7, help="Aux classifier loss weight")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        aux_weight=args.aux_weight,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_abs = cfg.output_dir.resolve()
    run_start = time.time()

    set_seed(cfg.seed)
    clear_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Output directory: {output_dir_abs}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Starting training for up to {cfg.max_epochs} epochs...")

    x_train, x_test, y_train, _ = read_data(cfg.data_dir)
    if y_train is None:
        raise RuntimeError("stage2_y_train.parquet is required for class-aware training.")

    x_train_fit, x_val_fit, y_train_fit, y_val_fit = train_test_split(
        x_train, y_train, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True, stratify=y_train
    )

    sampler = build_weighted_sampler(y_train_fit) if cfg.weighted_sampling else None
    class_weights_tensor = compute_class_weights_tensor(y_train_fit, n_classes=cfg.n_classes)

    print("Class distribution (train split):")
    uniq, cnt = np.unique(y_train_fit, return_counts=True)
    for c, n in zip(uniq, cnt):
        print(f"  class {int(c)}: {int(n):,}")
    print(f"Class weights: {[round(float(v), 4) for v in class_weights_tensor.tolist()]}")

    train_loader = make_loader(x_train_fit, cfg.batch_size, shuffle=True, y=y_train_fit, sampler=sampler)
    val_loader = make_loader(x_val_fit, cfg.batch_size, shuffle=False, y=y_val_fit)

    model = VAE(input_dim=x_train.shape[1], latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims).to(device)
    aux_clf = nn.Linear(cfg.latent_dim, cfg.n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_clf.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "train_aux": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
    }
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_path = cfg.output_dir / "vae_v3_16_best.pt"

    for epoch in range(cfg.max_epochs):
        beta = min(cfg.beta_target, cfg.beta_target * (epoch + 1) / max(1, cfg.warmup_epochs))
        train_metrics = run_epoch(
            model=model,
            aux_clf=aux_clf,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            free_bits=cfg.free_bits,
            use_amp=cfg.use_amp,
            train=True,
            class_weights_tensor=class_weights_tensor,
            aux_weight=cfg.aux_weight,
        )
        val_metrics = run_epoch(
            model=model,
            aux_clf=aux_clf,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            free_bits=cfg.free_bits,
            use_amp=cfg.use_amp,
            train=False,
            class_weights_tensor=class_weights_tensor,
            aux_weight=0.0,
        )

        scheduler.step(val_metrics["recon"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["train_aux"].append(train_metrics["aux"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])

        is_best = val_metrics["recon"] < best_val
        if val_metrics["recon"] < best_val:
            best_val = val_metrics["recon"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "aux_clf_state": aux_clf.state_dict(),
                    "val_recon": val_metrics["recon"],
                    "val_kl": val_metrics["kl"],
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        progress_pct = ((epoch + 1) / cfg.max_epochs) * 100.0
        best_tag = " [BEST]" if is_best else ""
        print(
            f"[EPOCH {epoch + 1:03d}/{cfg.max_epochs}] "
            f"{progress_pct:6.2f}% | "
            f"beta={beta:.3f} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"recon={val_metrics['recon']:.6f} | "
            f"kl={val_metrics['kl']:.6f} | "
            f"aux={train_metrics['aux']:.6f} | "
            f"patience={epochs_no_improve}/{cfg.patience}"
            f"{best_tag}"
        )

        if epochs_no_improve >= cfg.patience:
            print(f"[INFO] Early stopping at epoch {epoch + 1}.")
            break

    clear_cache()
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    aux_clf.load_state_dict(best_ckpt["aux_clf_state"])

    print(f"VAE-V3 best checkpoint: E{best_ckpt['epoch']}  val_recon={best_ckpt['val_recon']:.6f}  val_kl={best_ckpt['val_kl']:.4f}")
    print("Exporting latent features...")
    z_train = encode_all(model, x_train, device=device, batch_size=cfg.batch_size)
    z_test = encode_all(model, x_test, device=device, batch_size=cfg.batch_size)

    z_train_path = cfg.output_dir / "vae_v3_z_train.parquet"
    z_test_path = cfg.output_dir / "vae_v3_z_test.parquet"
    tsne_path = cfg.output_dir / "vae-v3_tsne.png"
    angle_path = cfg.output_dir / "vae-v3_angle_distributions.png"
    curves_path = cfg.output_dir / "vae-v3_training_curves.png"

    save_latents(z_train, z_train_path)
    print(f"  Saved {z_train_path.resolve()}  shape=torch.Size({list(z_train.shape)})")
    save_latents(z_test, z_test_path)
    print(f"  Saved {z_test_path.resolve()}  shape=torch.Size({list(z_test.shape)})")

    print("\nQuick checks:")
    min_dist = min_class_centroid_distance(z_train, y_train)
    if np.isfinite(min_dist):
        print(f"  [6/8] Min class centroid dist:    {min_dist:.4f}")
    print(f"  [7/8] Test angle diversity:       {'PASS' if angle_diversity_pass(z_test) else 'WARN'}")

    save_training_curves(history, curves_path)
    print(f"  Saved {curves_path.resolve()}")
    save_angle_distribution_plot(z_train, angle_path)
    print(f"  Saved {angle_path.resolve()}")
    save_tsne_plot(z_train, y_train, tsne_path, seed=cfg.seed)
    print(f"  Saved {tsne_path.resolve()}")

    latent_stats = {
        "z_train_shape": list(z_train.shape),
        "z_test_shape": list(z_test.shape),
        "z_train_min": float(z_train.min()),
        "z_train_max": float(z_train.max()),
        "z_train_mean": z_train.mean(axis=0).tolist(),
        "z_train_std": z_train.std(axis=0).tolist(),
        "min_class_centroid_distance": float(min_dist) if np.isfinite(min_dist) else None,
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

    elapsed = time.time() - run_start
    print(
        f"\n[vae_v3] Done in {int(elapsed)}s ({elapsed / 60.0:.1f} min) | "
        f"best E{best_epoch + 1}  val_recon={history['val_recon'][best_epoch]:.6f}"
    )
    print(f"[OK] Trained VAE v3 with latent_dim={cfg.latent_dim} on device={device}")
    print(f"[OK] Best epoch: {best_epoch + 1} | Best val_loss: {best_val:.6f}")
    print(f"[LINK] {z_train_path.resolve()}")
    print(f"[LINK] {z_test_path.resolve()}")
    print(f"[LINK] {curves_path.resolve()}")
    print(f"[LINK] {tsne_path.resolve()}")
    print(f"[LINK] {angle_path.resolve()}")
    print(f"[LINK] {(cfg.output_dir / 'vae_v3_16_config.json').resolve()}")
    print(f"[LINK] {(cfg.output_dir / 'vae_v3_16_training_history.json').resolve()}")


if __name__ == "__main__":
    main()
