"""GPU smoke test for LSA-NODE.

Runs the ODE block on an A100 to verify CUDA works, prints timing,
and exits. Intended for `srun --pty ... python gpu_smoke.py`.
"""

from __future__ import annotations

import time

import torch
from torchdiffeq import odeint, odeint_adjoint

from lsa_node.models.ode_func import ODEFunc
from lsa_node.models.stft_attention import stft


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-ok", action="store_true",
                   help="Run on CPU if no GPU is available (default: exit).")
    args = p.parse_args()

    print(f"torch {torch.__version__}  cuda={torch.version.cuda}  avail={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        if not args.cpu_ok:
            print("CUDA not available — pass --cpu-ok to run on CPU (slow). Exiting.")
            return
        print("CUDA not available — running on CPU (slow!)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Realistic sizes for our smoke config.
    B, N, T = 8, 64, 12
    n_fft, hop, d_model = 16, 4, 64

    func = ODEFunc(n_fft=n_fft, hop_length=hop, d_model=d_model, n_heads=4).to(device)

    # Build a dictionary (B, T*L, 2*n_freqs).
    z = torch.randn(B, T, N, device=device)
    H = stft(z.reshape(B * T, N), n_fft=n_fft, hop_length=hop)
    L = H.shape[1]
    H = H.reshape(B, T * L, -1)
    func.set_dictionary(H, H.clone(), signal_length=N)

    h0 = torch.randn(B, N, device=device)
    t_span = torch.linspace(0.0, 1.0, 5, device=device)

    # Forward (no grad) timing.
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = odeint(func, h0, t_span, method="dopri5", rtol=1e-3, atol=1e-4)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"odeint forward: {time.time() - t0:.3f}s   out.shape={tuple(out.shape)}")

    # Backward via adjoint.
    h0_req = h0.detach().requires_grad_(True)
    func.set_dictionary(H, H.clone(), signal_length=N)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = odeint_adjoint(func, h0_req, t_span, method="dopri5", rtol=1e-3, atol=1e-4)
    out.sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"odeint_adjoint fwd+bwd: {time.time() - t0:.3f}s   h0.grad norm={h0_req.grad.norm().item():.3e}")

    if device == "cuda":
        print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        print(f"device       : {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
