"""Tests for SlidingWindowDataset + collate."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from lsa_node.data import SlidingWindowDataset, collate_ts_items
from lsa_node.data.synthetic import periodic_3d


def _make_ds(in_len: int = 5, out_len: int = 3, **kwargs) -> SlidingWindowDataset:
    traj = periodic_3d(variant="A", n_points=40, t_max=10.0)
    return SlidingWindowDataset(traj, in_len=in_len, out_len=out_len, **kwargs)


def test_window_shapes_regular() -> None:
    ds = _make_ds()
    item = ds[0]
    assert item.x_obs.shape == (5, 3)
    assert item.t_obs.shape == (5,)
    assert item.x_query.shape == (3, 3)
    assert item.t_query.shape == (3,)


def test_window_count() -> None:
    ds = _make_ds(in_len=5, out_len=3, stride=8)
    # N=40, window=8, stride=8 → ⌊(40-8)/8⌋ + 1 = 5
    assert len(ds) == 5


def test_irregular_shrinks_input_segment() -> None:
    ds = _make_ds(irregular=True, irregular_keep_rate=0.3)
    item = ds[0]
    # On 5 points with rate 0.3 we expect roughly 1–4 kept (min 2 enforced).
    assert 2 <= item.x_obs.shape[0] <= 5


def test_collate_pads_to_max() -> None:
    ds = _make_ds(in_len=5, out_len=3, irregular=True, irregular_keep_rate=0.3)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_ts_items)
    batch = next(iter(loader))
    B, Tmax, in_dim = batch["x_obs"].shape
    assert B == 4 and in_dim == 3
    assert batch["mask_obs"].shape == (B, Tmax)
    assert batch["mask_obs"].any(dim=-1).all()              # every item has ≥1 valid obs
    assert batch["x_query"].shape == (B, 3, 3)
    assert batch["t_query"].shape == (3,)


def test_dataset_compatible_with_lsa_node_signature() -> None:
    from lsa_node.models import LSANODE

    ds = _make_ds(in_len=6, out_len=4)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_ts_items)
    batch = next(iter(loader))

    model = LSANODE(
        in_dim=3, out_dim=3, hidden_dim=32, n_fft=8, hop_length=4, d_att=16,
        n_heads=2, solver="rk4", use_adjoint=False,
    )
    y = model(batch["x_obs"], batch["t_obs"], batch["t_query"])
    assert y.shape == (4, 2, 3)
