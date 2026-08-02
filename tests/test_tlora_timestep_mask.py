"""Unit tests for T-LoRA timestep-dependent rank masking.

Covers the shared ``LoRAModule`` / ``LoRANetwork`` path used by Krea 2
(``networks.lora_krea2`` → ``networks.lora``). CPU-only; no DiT weights needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from musubi_tuner.networks.lora import LoRAModule, LoRANetwork, create_network


def _t(val: float, device: str = "cpu") -> torch.Tensor:
    return torch.tensor([val], device=device, dtype=torch.float32)


def _make_module(lora_dim: int = 32) -> LoRAModule:
    base = nn.Linear(8, 8, bias=False)
    return LoRAModule("test", base, multiplier=1.0, lora_dim=lora_dim, alpha=lora_dim)


def _make_network(ranks: list[int], *, min_rank: int = 1, alpha_rank_scale: float = 1.0) -> LoRANetwork:
    """Minimal LoRANetwork stub with synthetic modules (no real DiT walk)."""
    net = LoRANetwork.__new__(LoRANetwork)
    nn.Module.__init__(net)
    net.use_timestep_mask = True
    net.min_rank = min_rank
    net.alpha_rank_scale = alpha_rank_scale
    net.lora_dim = ranks[0]
    net.text_encoder_loras = []
    net.unet_loras = [_make_module(r) for r in ranks]
    return net


def test_default_mask_is_identity():
    m = _make_module(16)
    assert m._timestep_mask.shape == (1, 16)
    assert torch.allclose(m._timestep_mask, torch.ones(1, 16))


def test_apply_timestep_mask_linear():
    m = _make_module(4)
    m._timestep_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lx = torch.randn(2, 4)
    out = m._apply_timestep_mask(lx)
    assert out.shape == lx.shape
    assert torch.allclose(out[:, :2], lx[:, :2])
    assert torch.allclose(out[:, 2:], torch.zeros_like(out[:, 2:]))


def test_apply_timestep_mask_conv2d_broadcast():
    m = _make_module(4)
    m._timestep_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    lx = torch.randn(2, 4, 3, 3)
    out = m._apply_timestep_mask(lx)
    assert out.shape == lx.shape
    assert torch.allclose(out[:, 0], lx[:, 0])
    assert torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]))
    assert torch.allclose(out[:, 2], lx[:, 2])
    assert torch.allclose(out[:, 3], torch.zeros_like(out[:, 3]))


def test_set_timestep_mask_high_noise_uses_min_rank():
    net = _make_network([32], min_rank=1)
    net.set_timestep_mask(_t(1000.0), max_timestep=1000.0)
    mask = net.unet_loras[0]._timestep_mask
    assert mask.sum().item() == 1  # floor = min_rank


def test_set_timestep_mask_low_noise_uses_full_rank():
    net = _make_network([32], min_rank=1)
    net.set_timestep_mask(_t(0.0), max_timestep=1000.0)
    mask = net.unet_loras[0]._timestep_mask
    assert mask.sum().item() == 32


def test_set_timestep_mask_midpoint():
    net = _make_network([32], min_rank=0)
    net.set_timestep_mask(_t(500.0), max_timestep=1000.0)
    # frac = 0.5 → r = 16
    assert net.unet_loras[0]._timestep_mask.sum().item() == 16


def test_mixed_rank_groups_share_per_rank_mask():
    net = _make_network([32, 32, 16, 16], min_rank=1)
    net.set_timestep_mask(_t(500.0), max_timestep=1000.0)
    a, b, c, d = net.unet_loras
    assert a._timestep_mask is b._timestep_mask
    assert c._timestep_mask is d._timestep_mask
    assert a._timestep_mask is not c._timestep_mask
    assert a._timestep_mask.shape == (1, 32)
    assert c._timestep_mask.shape == (1, 16)


def test_min_rank_clamped_into_small_groups():
    """min_rank above a group's own rank must not break the small group."""
    net = _make_network([32, 8], min_rank=16)
    net.set_timestep_mask(_t(1000.0), max_timestep=1000.0)
    m32, m8 = (m._timestep_mask for m in net.unet_loras)
    assert m32.sum().item() == 16
    assert m8.sum().item() == 8  # clamped floor = rank


def test_clear_timestep_mask_restores_ones():
    net = _make_network([32], min_rank=1)
    net.set_timestep_mask(_t(1000.0), max_timestep=1000.0)
    assert net.unet_loras[0]._timestep_mask.sum().item() < 32
    net.clear_timestep_mask()
    assert torch.allclose(net.unet_loras[0]._timestep_mask, torch.ones(1, 32))


def test_clear_timestep_mask_safe_before_set():
    net = _make_network([8])
    net.clear_timestep_mask()  # must not raise


def test_disabled_flag_is_noop():
    net = _make_network([32])
    net.use_timestep_mask = False
    before = net.unet_loras[0]._timestep_mask.clone()
    net.set_timestep_mask(_t(1000.0), max_timestep=1000.0)
    assert torch.allclose(net.unet_loras[0]._timestep_mask, before)


def test_create_network_parses_tlora_args():
    # Tiny fake unet with one Linear so create_network can walk it.
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

    unet = Tiny()
    network = create_network(
        None,  # all Linear
        "lora_unet",
        1.0,
        4,
        4.0,
        None,
        [],
        unet,
        use_timestep_mask="True",
        min_rank="2",
        alpha_rank_scale="0.5",
    )
    assert network.use_timestep_mask is True
    assert network.min_rank == 2
    assert network.alpha_rank_scale == 0.5
    assert len(network.unet_loras) == 1
    assert "_timestep_mask" not in network.state_dict()  # persistent=False


def test_forward_respects_mask():
    """With a partial mask, the LoRA delta must shrink vs full-rank."""
    torch.manual_seed(0)
    base = nn.Linear(8, 8, bias=False)
    lora = LoRAModule("blk", base, multiplier=1.0, lora_dim=4, alpha=4)
    nn.init.normal_(lora.lora_up.weight, std=0.1)
    lora.apply_to()
    lora.train()
    x = torch.randn(2, 8)

    lora._timestep_mask.fill_(1.0)
    full = base(x).detach()

    lora._timestep_mask.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    masked = base(x).detach()

    # Both differ from the frozen base alone (up weights nonzero), and from each other.
    org = lora.org_forward(x)
    assert not torch.allclose(full, org)
    assert not torch.allclose(masked, org)
    assert not torch.allclose(full, masked)
