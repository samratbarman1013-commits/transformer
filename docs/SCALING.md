# Scaling plan: 1M → ~500M

The code does not change; only the config, the data, and the compute do.

## Config comparison

| | `1m_prototype` | `500m` |
|---|---|---|
| params (approx.) | 1.05M | 516M |
| layers | 4 | 30 |
| hidden dim | 128 | 1152 |
| heads | 4 | 18 |
| context | 256 | 1024 |
| vocab (BPE) | 2,048 | 32,768 |
| hardware | 1 GPU / free Colab / CPU | multi-GPU node |
| precision | fp32 | bf16 + FSDP |
| training tokens | ~50M (overkill is fine) | 10–25B |

## What actually has to change for the 500M run

1. **Data.** Chinchilla-style guidance says ~20 tokens per parameter:
   ~10B tokens minimum, ideally 25B. TinyStories-class corpora exist at this
   scale but need mixing and filtering. Data quality will matter more than
   any hyperparameter.

2. **Distributed training.** The current loop must be wrapped in PyTorch FSDP
   (or DDP + ZeRO). The model code itself is already standard — the changes
   are confined to `model/train.py`: the sampler, the optimizer setup, and
   checkpoint sharding. Mixed precision (bf16) is a `torch.autocast` wrapper.

3. **Tokenizer.** Retrain the BPE at 32,768 vocab on the full corpus
   (`config/500m.yaml` already declares it).

4. **Inference.** Serve quantized: int8 via `bitsandbytes`, or 4-bit groupwise
   GGUF for on-device use. A 500M model at 4-bit is ~250MB — comfortably
   phone-sized, which is the whole point of stopping at 500M.

## Compute reality check

- 1M prototype on TinyStories (~50M tokens): an hour or two on a free Colab
  GPU; the overfit smoke test runs on CPU in minutes.
- 500M on 10–25B tokens: think multi-GPU (8×A100 class) for days-to-weeks —
  this is a real project, not a weekend one. Renting equivalent spot capacity
  is a four-to-five-figure-dollar decision. **Ask before spending.**

## Why 500M is a good target

Small enough to (a) train on rented/commodity compute, (b) quantize to
phone-sized, (c) read and understand fully. Large enough to follow
instructions and use tools with consistent formatting. The capability gap to
big models is mostly knowledge, which the tool system (search, code) is there
to cover.
