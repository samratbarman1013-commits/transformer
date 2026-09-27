"""Tests that need torch. CI installs the CPU build; locally they skip if absent."""
import pytest

torch = pytest.importorskip("torch")

from model.config import ModelConfig, estimate_params
from model.transformer import GPT

PROTO = ModelConfig(n_layer=4, n_embd=128, n_head=4, block_size=256, vocab_size=2048)


def test_param_budget_1m():
    model = GPT(PROTO)
    n = model.num_params()
    assert 700_000 <= n <= 1_400_000, f"prototype is {n:,} params, outside the ~1M budget"
    assert abs(estimate_params(PROTO) - n) < 60_000, "estimate_params drifted from reality"


def test_forward_shapes():
    model = GPT(PROTO)
    x = torch.randint(0, PROTO.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, PROTO.vocab_size)
    assert torch.isfinite(loss)


def test_causality_future_tokens_do_not_affect_past():
    """Change the last token; the first-position logits must not move."""
    torch.manual_seed(0)
    model = GPT(PROTO).eval()
    a = torch.randint(0, PROTO.vocab_size, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % PROTO.vocab_size
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[0, :4], lb[0, :4]), "attention leaked future information"


def test_generate_respects_length_and_context():
    model = GPT(PROTO).eval()
    x = torch.randint(0, PROTO.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=10, temperature=0.0)
    assert out.shape == (1, 14)
