"""Tokenizer utilities: build a small ByteLevel BPE, or load an existing one.

The prototype trains its own tokenizer so the vocabulary size (and therefore
the embedding parameter budget) is ours to choose — 2,048 for the 1M model.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>"]


def build_bpe(
    data_paths: list[str | Path],
    vocab_size: int,
    out_path: str | Path,
) -> Tokenizer:
    """Train a ByteLevel BPE tokenizer over the given text files and save it."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train(files=[str(p) for p in data_paths], trainer=trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def ensure_tokenizer(
    tokenizer_path: str | Path,
    data_paths: list[str | Path],
    vocab_size: int,
) -> tuple[Tokenizer, int]:
    """Load the tokenizer at tokenizer_path, or build it from data if missing.

    Returns (tokenizer, actual_vocab_size). The actual size may differ slightly
    from the requested size; callers should trust the returned value.
    """
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        tok = load_tokenizer(tokenizer_path)
    else:
        tok = build_bpe(data_paths, vocab_size, tokenizer_path)
    return tok, tok.get_vocab_size(with_added_tokens=True)
