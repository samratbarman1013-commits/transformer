# Data

Put training text files here.

- `smoke.txt` — tiny built-in corpus for the overfit smoke test (committed).
- `corpus.txt` — your real corpus; downloaded, never committed (see .gitignore).

## Getting TinyStories (recommended first corpus)

```bash
# HuggingFace datasets CLI
pip install datasets
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
text = "\n\n".join(t["text"] for t in ds.select(range(200_000)))  # ~180MB
open("data/corpus.txt", "w").write(text)
EOF
```

200k stories ≈ 45M tokens — comfortably beyond what a 1M-parameter model can
memorize, which is what you want.

## Training with it

```bash
python -m model.train --config config/1m_prototype.yaml --data data/corpus.txt
```

The tokenizer (2,048-token BPE) is built automatically from the corpus on the
first run and saved to `out/1m-prototype/tokenizer.json`.
