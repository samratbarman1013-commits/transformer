"""Generate text from a trained checkpoint.

Usage:
    python -m model.sample --config config/1m_prototype.yaml --prompt "once upon a time"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ExperimentConfig, ModelConfig
from .tokenizer import load_tokenizer
from .transformer import GPT


def load_model(ckpt_path: Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Rebuild the model from a checkpoint (config is stored inside it)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a checkpoint")
    parser.add_argument("--config", required=True, help="config used to find out/<name>/ckpt.pt")
    parser.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --config lookup)")
    parser.add_argument("--prompt", default="", help="starting text")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path("out") / cfg.name / "ckpt.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} — run model.train first")

    model, _ckpt = load_model(ckpt_path, args.device)
    tok = load_tokenizer(Path("out") / cfg.name / "tokenizer.json")

    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    y = model.generate(x, args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
