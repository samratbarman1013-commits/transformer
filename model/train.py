"""Training entry point.

Usage:
    python -m model.train --config config/1m_prototype.yaml --data data/corpus.txt
    python -m model.train --config config/1m_prototype.yaml --data data/smoke.txt --smoke

--smoke runs a short overfit run on a tiny file: after a minute or two on CPU the
loss should approach zero and generation should reproduce the file. That is
deliberate — it proves the architecture and loop work before any real training.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .tokenizer import ensure_tokenizer
from .transformer import GPT


def get_lr(it: int, train) -> float:
    """Warmup followed by cosine decay to min_lr."""
    if it < train.warmup_iters:
        return train.learning_rate * (it + 1) / (train.warmup_iters + 1)
    if it > train.lr_decay_iters:
        return train.min_lr
    ratio = (it - train.warmup_iters) / (train.lr_decay_iters - train.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return train.min_lr + coeff * (train.learning_rate - train.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, train_cfg, batch_size: int, device):
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            ix = torch.randint(len(data) - model.cfg.block_size, (batch_size,))
            x = torch.stack([data[i : i + model.cfg.block_size] for i in ix])
            y = torch.stack([data[i + 1 : i + 1 + model.cfg.block_size] for i in ix])
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def encode_file(path: Path, tok) -> np.ndarray:
    ids = tok.encode(path.read_text(encoding="utf-8", errors="ignore")).ids
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model defined by a config file")
    parser.add_argument("--config", required=True, help="path to a YAML config, e.g. config/1m_prototype.yaml")
    parser.add_argument("--data", required=True, help="path to the training text file")
    parser.add_argument("--val-frac", type=float, default=0.05, help="fraction of data held out for val loss")
    parser.add_argument("--out-dir", default="out", help="where checkpoints are written")
    parser.add_argument("--smoke", action="store_true", help="short overfit run to validate the pipeline")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(args.data)
    out_dir = Path(args.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer: build from this corpus if one does not exist yet.
    tok_path = out_dir / "tokenizer.json"
    tok, actual_vocab = ensure_tokenizer(tok_path, [data_path], cfg.model.vocab_size)
    if actual_vocab != cfg.model.vocab_size:
        print(f"[info] vocab_size: config said {cfg.model.vocab_size}, tokenizer has {actual_vocab}; using actual")
        cfg.model.vocab_size = actual_vocab

    # 2. Data: encode once, split train/val.
    ids = encode_file(data_path, tok)
    split = max(1, int(len(ids) * (1 - args.val_frac)))
    train_data = torch.from_numpy(ids[:split].astype(np.int64))
    val_data = torch.from_numpy(ids[split:].astype(np.int64))
    print(f"[data] {len(ids):,} tokens (train {len(train_data):,} / val {len(val_data):,})")
    if len(ids) < cfg.model.block_size + 2:
        raise SystemExit(f"dataset too small ({len(ids)} tokens) for block_size {cfg.model.block_size}")

    max_iters = cfg.train.max_iters
    if args.smoke:
        max_iters, cfg.train.eval_interval = 200, 50

    # 3. Model + optimizer.
    model = GPT(cfg.model).to(device)
    print(f"[model] {model.num_params():,} params ({model.num_params(non_embedding=False):,} incl. embeddings)")
    optimizer = model.configure_optimizers(
        weight_decay=cfg.train.weight_decay,
        learning_rate=cfg.train.learning_rate,
        betas=(cfg.train.beta1, cfg.train.beta2),
        device_type=device,
    )

    def get_batch(split_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(split_data) - cfg.model.block_size, (cfg.train.batch_size,))
        x = torch.stack([split_data[i : i + cfg.model.block_size] for i in ix])
        y = torch.stack([split_data[i + 1 : i + 1 + cfg.model.block_size] for i in ix])
        return x.to(device), y.to(device)

    # 4. The loop.
    t0 = time.time()
    best_val = float("inf")
    for it in range(max_iters):
        lr = get_lr(it, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.train.eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg.train, cfg.train.batch_size, device)
            dt = time.time() - t0
            print(
                f"iter {it:5d}/{max_iters} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {dt:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "iter": it, "val_loss": best_val},
                    out_dir / "ckpt.pt",
                )

        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

    print(f"done. best val loss {best_val:.4f}; checkpoint at {out_dir / 'ckpt.pt'}")
    if args.smoke:
        print("smoke test passed if val loss dropped close to 0 (tiny file, model memorizes it)")


if __name__ == "__main__":
    main()
