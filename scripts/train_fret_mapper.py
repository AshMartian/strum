"""train_fret_mapper.py — Train MLP fret-mapper on cached BP→GT dataset.

Reads .npz files from --cache-dir. Each file contains:
  X: (N, 95) per-onset features
  Y: (N, 5)  binary fret labels [G,R,Y,B,O]

Trains a small MLP with 5 sigmoid heads (BCE loss). Splits 90/10 by song
(not by sample) to avoid leakage. Saves model + scaler to --out.

Usage:
  python scripts/train_fret_mapper.py \
      --cache-dir /mnt/ml-data/fret_mapper_cache \
      --out checkpoints/fret_mapper_v1.pt \
      --epochs 30 --batch 4096 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.fret_mapper import FretMapperMLP  # noqa: E402

log = logging.getLogger("train_fret_mapper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_cache(
    cache_dir: Path,
    val_frac: float = 0.10,
    seed: int = 42,
    *,
    use_catalog_splits: bool = False,
):
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No npz files in {cache_dir}")
    if use_catalog_splits:
        split_files: dict[str, list[Path]] = {"train": [], "val": []}
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                raw_split = data.get("split")
                if raw_split is None:
                    raise SystemExit(f"Catalog cache file is missing split metadata: {path.name}")
                split = str(raw_split.item())
            if split in split_files:
                split_files[split].append(path)
        train_files = split_files["train"]
        val_files = split_files["val"]
        if not train_files or not val_files:
            raise SystemExit("Catalog cache requires at least one train and one val song")
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        val_files = files[:n_val]
        train_files = files[n_val:]

    def cat(fs):
        Xs, Ys = [], []
        for f in fs:
            d = np.load(f, allow_pickle=True)
            Xs.append(d["X"])
            Ys.append(d["Y"])
        return np.concatenate(Xs, 0), np.concatenate(Ys, 0)

    Xtr, Ytr = cat(train_files)
    Xva, Yva = cat(val_files)
    return Xtr, Ytr, Xva, Yva, train_files, val_files


def evaluate(model, X, Y, device, batch=8192):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            ps.append(p)
    P = np.concatenate(ps, 0)
    pred = (P >= 0.5).astype(np.float32)
    # Per-fret precision/recall/f1
    out = {}
    for k, name in enumerate("GRYBO"):
        tp = float(((pred[:, k] == 1) & (Y[:, k] == 1)).sum())
        fp = float(((pred[:, k] == 1) & (Y[:, k] == 0)).sum())
        fn = float(((pred[:, k] == 0) & (Y[:, k] == 1)).sum())
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[name] = (prec, rec, f1)
    # Subset accuracy (all 5 frets correct)
    out["exact"] = float((pred == Y).all(axis=1).mean())
    out["mean_f1"] = float(np.mean([out[c][2] for c in "GRYBO"]))
    out["chord_frac_pred"] = float((pred.sum(1) >= 2).mean())
    out["chord_frac_gt"] = float((Y.sum(1) >= 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--use-catalog-splits",
        action="store_true",
        help="Require per-cache-file train/val splits emitted by a catalog task view.",
    )
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument(
        "--metrics-out",
        type=Path,
        help="Optional path for a JSON-safe training summary owned by a worker.",
    )
    ap.add_argument(
        "--pos-weight-cap",
        type=float,
        default=5.0,
        help="Cap on per-class positive weight. Set 1.0 to disable.",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    mps_backend = getattr(torch.backends, "mps", None)
    if device == "mps" and (mps_backend is None or not mps_backend.is_available()):
        raise SystemExit("MPS was requested but is unavailable")
    log.info(f"Device: {device}")

    Xtr, Ytr, Xva, Yva, tr_files, va_files = load_cache(
        args.cache_dir,
        args.val_frac,
        args.seed,
        use_catalog_splits=args.use_catalog_splits,
    )
    log.info(f"Train: {Xtr.shape}  ({len(tr_files)} songs)")
    log.info(f"Val:   {Xva.shape}  ({len(va_files)} songs)")
    log.info(
        f"GT chord_frac train={(Ytr.sum(1) >= 2).mean():.2%}  val={(Yva.sum(1) >= 2).mean():.2%}"
    )

    # Standardize features (fit on train)
    mu = Xtr.mean(0).astype(np.float32)
    sd = Xtr.std(0).astype(np.float32) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    # Class weights for positive frets (slight upweight to push chord_fraction)
    pos_freq = Ytr.mean(0)
    pos_weight = torch.tensor(
        np.minimum(args.pos_weight_cap, (1.0 - pos_freq) / np.maximum(pos_freq, 1e-3)),
        dtype=torch.float32,
        device=device,
    )
    log.info(f"pos_weight per fret: {pos_weight.cpu().numpy().round(2).tolist()}")

    model = FretMapperMLP(
        in_dim=Xtr.shape[1], hidden=args.hidden, out_dim=5, p_drop=args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr)
    ds = TensorDataset(Xtr_t, Ytr_t)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device == "cuda")
    )

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * xb.size(0)
        sched.step()
        avg = total / len(ds)
        m = evaluate(model, Xva, Yva, device)
        msg = (
            f"ep {epoch:2d}/{args.epochs}  loss={avg:.4f}  "
            f"val_mean_f1={m['mean_f1']:.3f}  exact={m['exact']:.3f}  "
            f"chord_frac p/g={m['chord_frac_pred']:.2f}/{m['chord_frac_gt']:.2f}  "
            f"({time.time() - t0:.1f}s)"
        )
        log.info(msg)
        for c in "GRYBO":
            p, r, f1 = m[c]
            log.info(f"   {c}: P={p:.2f} R={r:.2f} F1={f1:.2f}")
        if m["mean_f1"] > best_f1:
            best_f1 = m["mean_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "strum-fret-mapper-weights/v1",
            "model_state": best_state,
            "input_dimension": Xtr.shape[1],
            "hidden": args.hidden,
            "output_dimension": 5,
            # Keep this payload compatible with ``torch.load(weights_only=True)``.
            # NumPy arrays would require unsafe pickle global allowlisting.
            "feature_mean": torch.from_numpy(mu.copy()),
            "feature_std": torch.from_numpy(sd.copy()),
        },
        args.out,
    )
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(
                {
                    "best_val_f1": float(best_f1),
                    "train_song_count": len(tr_files),
                    "val_song_count": len(va_files),
                    "feature_dimension": int(Xtr.shape[1]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    log.info(f"Saved best model (val_mean_f1={best_f1:.3f}) → {args.out}")


if __name__ == "__main__":
    main()
