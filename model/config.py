"""Experiment configuration.

Everything tunable lives in YAML (see config/*.yaml) so scaling from the
1M-parameter prototype to the ~500M model is a config change, not a rewrite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    block_size: int = 256
    vocab_size: int = 2048
    dropout: float = 0.0
    bias: bool = False


@dataclass
class TrainConfig:
    max_iters: int = 5000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 3e-5
    eval_interval: int = 250
    eval_iters: int = 20
    seed: int = 1337


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def load(path: Union[str, Path]) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        model = ModelConfig(**raw.get("model", {}))
        train = TrainConfig(**raw.get("train", {}))
        return ExperimentConfig(name=raw.get("name", Path(path).stem), model=model, train=train)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def estimate_params(cfg: ModelConfig) -> int:
    """Approximate parameter count (tied embeddings, LayerNorm weights ignored)."""
    per_layer = 4 * cfg.n_embd**2 + 8 * cfg.n_embd**2  # attention qkv+proj, then MLP
    return cfg.vocab_size * cfg.n_embd + cfg.n_layer * per_layer
