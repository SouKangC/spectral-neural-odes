"""Round-trip and shape tests for STFT/ISTFT + conjugate-symmetry projection.

The frame representation is the half spectrum split into real/imag halves:
last-dim length is ``2 * (n_fft // 2 + 1) = n_fft + 2``. See
``code/lsa_node/models/stft_attention.py`` for the rationale.
"""

from __future__ import annotations

import pytest
import torch

from lsa_node.models.stft_attention import (
    conjugate_symmetry_projection,
    istft,
    stft,
)

torch.manual_seed(0)


def _channel_dim(n_fft: int) -> int:
    return 2 * (n_fft // 2 + 1)


# ---------------------------------------------------------------------------
# Shape and reconstruction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_fft,hop", [(16, 4), (32, 8), (8, 4), (64, 16)])
def test_stft_shape(n_fft: int, hop: int) -> None:
    N = 256
    x = torch.randn(3, N)
    H = stft(x, n_fft=n_fft, hop_length=hop)
    assert H.shape[0] == 3
    assert H.shape[-1] == _channel_dim(n_fft)
    # L = floor(N / hop) + 1 when center=True (matches torch.stft default).
    expected_L = N // hop + 1
    assert H.shape[-2] == expected_L


@pytest.mark.parametrize("n_fft,hop", [(16, 4), (32, 8), (8, 4)])
def test_stft_istft_roundtrip(n_fft: int, hop: int) -> None:
    """ISTFT inverts STFT for real signals (up to numerical noise)."""
    N = 256
    x = torch.randn(2, N)
    H = stft(x, n_fft=n_fft, hop_length=hop)
    H = conjugate_symmetry_projection(H, n_fft=n_fft)  # idempotent for STFT of real x
    x_hat = istft(H, n_fft=n_fft, hop_length=hop, length=N)
    assert x_hat.shape == x.shape
    assert torch.allclose(x, x_hat, atol=1e-5), (x - x_hat).abs().max().item()


def test_conjugate_symmetry_projection_zeros_correct_entries() -> None:
    """DC.imag must be zero after projection; Nyquist.imag too if n_fft is even."""
    n_fft, L = 16, 8
    cdim = _channel_dim(n_fft)
    h = torch.randn(4, L, cdim)
    h_sym = conjugate_symmetry_projection(h, n_fft)

    n_freqs = n_fft // 2 + 1
    assert torch.all(h_sym[..., n_freqs] == 0.0)        # DC imag
    if n_fft % 2 == 0:
        assert torch.all(h_sym[..., -1] == 0.0)         # Nyquist imag


def test_istft_of_random_projected_spectrum_is_real() -> None:
    """Random h_info, projected, ISTFT runs and produces a real signal."""
    n_fft, hop, N = 16, 4, 256
    cdim = _channel_dim(n_fft)
    L = N // hop + 1
    h = torch.randn(4, L, cdim)
    h_sym = conjugate_symmetry_projection(h, n_fft)
    out = istft(h_sym, n_fft=n_fft, hop_length=hop, length=N)
    assert out.shape == (4, N)
    assert torch.is_floating_point(out) and not out.is_complex()
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# The d = N symmetry test (task 2.8b in progress.md).
# ---------------------------------------------------------------------------
def test_encoder_and_ode_step_stft_have_same_frame_shape() -> None:
    """Encoder-side STFT (over a length-N patch z_{t_i}) and ODE-step STFT
    (over the length-N hidden state h(t)) must produce frames of identical
    shape under d = N. See docs/qa_design_choices.md Q2."""
    N, n_fft, hop = 64, 16, 4
    z = torch.randn(N)
    h = torch.randn(N)
    Hz = stft(z, n_fft=n_fft, hop_length=hop)
    Hh = stft(h, n_fft=n_fft, hop_length=hop)
    assert Hz.shape == Hh.shape


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
def test_stft_istft_gradients_flow() -> None:
    """End-to-end ``x → STFT → mask → ISTFT → loss`` is differentiable."""
    n_fft, hop, N = 16, 4, 128
    x = torch.randn(2, N, requires_grad=True)
    H = stft(x, n_fft=n_fft, hop_length=hop)
    H = conjugate_symmetry_projection(H, n_fft)
    x_hat = istft(H, n_fft=n_fft, hop_length=hop, length=N)
    loss = (x_hat - x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
