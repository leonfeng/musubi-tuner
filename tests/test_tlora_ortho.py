"""Unit tests for Orthogonal T-LoRA (SVD Q/P/λ + frozen baseline).

CPU-only; no DiT weights needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from musubi_tuner.networks.lora import OrthoTLoRAModule, create_network


def _make_ortho(in_dim=8, out_dim=8, r=4, *, sig_type="last", ortho_init="random"):
    torch.manual_seed(0)
    base = nn.Linear(in_dim, out_dim, bias=False)
    return OrthoTLoRAModule(
        "blk",
        base,
        multiplier=1.0,
        lora_dim=r,
        alpha=r,
        sig_type=sig_type,
        ortho_init=ortho_init,
    ), base


def test_ortho_init_zero_contribution():
    """Trainable and frozen copies match at init → delta is ~0."""
    lora, base = _make_ortho()
    lora.apply_to()
    lora.train()
    x = torch.randn(2, 8)
    org = lora.org_forward(x)
    out = base(x)
    assert torch.allclose(out, org, atol=1e-5, rtol=1e-4)


def test_ortho_base_layer_init_zero_contribution():
    lora, base = _make_ortho(ortho_init="base_layer", sig_type="principal")
    lora.apply_to()
    x = torch.randn(2, 8)
    assert torch.allclose(base(x), lora.org_forward(x), atol=1e-5, rtol=1e-4)


def test_ortho_frozen_not_trainable():
    lora, _ = _make_ortho()
    assert not any(p.requires_grad for p in lora.base_A.parameters())
    assert not any(p.requires_grad for p in lora.base_B.parameters())
    assert lora.lora_lambda.requires_grad
    assert lora.lora_down.weight.requires_grad
    assert lora.lora_up.weight.requires_grad


def test_ortho_forward_diverges_after_drift():
    lora, base = _make_ortho()
    lora.apply_to()
    lora.train()
    with torch.no_grad():
        lora.lora_lambda.add_(0.5)
    x = torch.randn(2, 8)
    org = lora.org_forward(x)
    out = base(x)
    assert not torch.allclose(out, org, atol=1e-4)


def test_ortho_respects_timestep_mask():
    lora, base = _make_ortho()
    lora.apply_to()
    lora.train()
    with torch.no_grad():
        lora.lora_up.weight.normal_(std=0.1)
        lora.lora_lambda.fill_(1.0)
    x = torch.randn(2, 8)

    lora._timestep_mask.fill_(1.0)
    full = base(x).detach()

    lora._timestep_mask.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    masked = base(x).detach()
    assert not torch.allclose(full, masked)


def test_distill_produces_standard_keys():
    lora, _ = _make_ortho()
    with torch.no_grad():
        lora.lora_lambda.add_(0.25)
        lora.lora_up.weight.add_(0.01)
    weights = lora.distill_to_lora_weights(dtype=torch.float32)
    assert set(weights) == {"lora_down.weight", "lora_up.weight", "alpha"}
    assert weights["lora_down.weight"].shape == (4, 8)
    assert weights["lora_up.weight"].shape == (8, 4)


def test_distill_matches_delta_approximately():
    """Distilled LoRA should approximate the ortho delta at identity mask."""
    lora, base = _make_ortho(in_dim=6, out_dim=5, r=3)
    lora.apply_to()
    with torch.no_grad():
        lora.lora_lambda.add_(torch.tensor([[0.3, -0.2, 0.5]]))
        lora.lora_down.weight.add_(0.05)
        lora.lora_up.weight.add_(0.05)

    x = torch.randn(4, 6)
    ortho_out = base(x).detach()
    org = lora.org_forward(x).detach()
    delta = ortho_out - org

    w = lora.distill_to_lora_weights(dtype=torch.float32)
    # Reconstruct delta from distilled factors (scale = alpha/r = 1).
    distilled_delta = x @ w["lora_down.weight"].T @ w["lora_up.weight"].T
    # Rank-r truncated SVD of a rank-≤2r matrix — allow moderate error.
    rel = (distilled_delta - delta).norm() / delta.norm().clamp(min=1e-8)
    assert rel.item() < 0.35


def test_create_network_tlora_ortho_parses_args():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)

    network = create_network(
        None,
        "lora_unet",
        1.0,
        2,
        2.0,
        None,
        [],
        Tiny(),
        tlora_ortho="True",
        sig_type="principal",
        ortho_init="base_layer",
        min_rank="1",
    )
    assert network.tlora_ortho is True
    assert network.use_timestep_mask is True  # auto-enabled
    assert network.sig_type == "principal"
    assert network.ortho_init == "base_layer"
    assert len(network.unet_loras) == 1
    assert isinstance(network.unet_loras[0], OrthoTLoRAModule)


def test_save_weights_distills_to_standard_lora():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)

    network = create_network(
        None,
        "lora_unet",
        1.0,
        2,
        2.0,
        None,
        [],
        Tiny(),
        tlora_ortho="True",
        sig_type="last",
        ortho_init="random",
    )
    network.apply_to(None, Tiny(), apply_text_encoder=False, apply_unet=True)
    # Drift trainable path so distill is non-trivial.
    with torch.no_grad():
        network.unet_loras[0].lora_lambda.add_(0.1)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ortho.safetensors"
        network.save_weights(str(path), torch.float32, metadata=None)
        from safetensors.torch import load_file

        sd = load_file(str(path))
    keys = list(sd.keys())
    assert any(k.endswith("lora_down.weight") for k in keys)
    assert any(k.endswith("lora_up.weight") for k in keys)
    assert not any("lora_lambda" in k for k in keys)
    assert not any("base_A" in k or "base_B" in k or "base_lambda" in k for k in keys)


def test_prepare_optimizer_skips_frozen():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)

    network = create_network(
        None,
        "lora_unet",
        1.0,
        2,
        2.0,
        None,
        [],
        Tiny(),
        tlora_ortho="True",
    )
    groups, _ = network.prepare_optimizer_params(unet_lr=1e-4)
    params = [p for g in groups for p in g["params"]]
    for lora in network.unet_loras:
        for p in list(lora.base_A.parameters()) + list(lora.base_B.parameters()):
            assert all(p is not q for q in params)
            assert not p.requires_grad
    assert any(p is network.unet_loras[0].lora_lambda for p in params)
