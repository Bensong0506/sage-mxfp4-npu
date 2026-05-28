"""Hadamard transform for Route B, implemented as matmul by a cached
Sylvester/Kronecker Hadamard matrix (equivalent to the normalized
Walsh-Hadamard transform used in SageAttention3's prototype).

For head_dim D (power of 2):  hadamard(x)[..., :] = (x @ H_D) / sqrt(D)
where H_D is the symmetric ±1 Hadamard matrix. Runs on NPU via torch.matmul
(Cube unit) — no custom kernel needed.
"""
import torch

_H_CACHE = {}


def _sylvester(D, device, dtype):
    """Build the natural-ordering (Sylvester) Hadamard matrix H_D, ±1 entries.
    Matches the recursive butterfly FWHT: H_{2n} = [[H_n, H_n], [H_n, -H_n]]."""
    key = (D, str(device), str(dtype))
    if key in _H_CACHE:
        return _H_CACHE[key]
    assert (D & (D - 1)) == 0, f"D={D} must be a power of 2"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    n = 1
    while n < D:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
        n *= 2
    _H_CACHE[key] = H
    return H


def hadamard_transform_npu(x):
    """Normalized Walsh-Hadamard transform over the last dim of x."""
    D = x.shape[-1]
    orig_shape = x.shape
    # Do the matmul in fp32 for accuracy, cast back.
    xf = x.reshape(-1, D).float()
    H = _sylvester(D, x.device, torch.float32)
    out = (xf @ H) / (D ** 0.5)
    return out.reshape(orig_shape).to(x.dtype)


# ---- Reference FWHT (numpy butterfly) for validation ----
def _fwht_ref(x_row):
    import numpy as np
    a = x_row.astype(np.float64).copy()
    D = a.shape[0]
    h = 1
    while h < D:
        for i in range(0, D, 2 * h):
            for j in range(i, i + h):
                u, v = a[j], a[j + h]
                a[j] = u + v
                a[j + h] = u - v
        h *= 2
    return a / (D ** 0.5)


if __name__ == "__main__":
    import torch_npu  # noqa: F401
    import numpy as np

    for D in (64, 128):
        # 1. Orthogonality: H @ H^T == D * I
        H = _sylvester(D, "cpu", torch.float32)
        ortho = torch.allclose(H @ H.t(), D * torch.eye(D), atol=1e-4)
        print(f"D={D} orthogonal(H@H^T==D*I): {ortho}")

        # 2. matmul impl vs numpy butterfly reference (ordering check)
        x = torch.randn(8, D, dtype=torch.float32)
        got = hadamard_transform_npu(x).numpy()
        ref = np.stack([_fwht_ref(x[r].numpy()) for r in range(x.shape[0])])
        max_diff = np.abs(got - ref).max()
        print(f"D={D} matmul vs butterfly max_diff={max_diff:.3e} "
              f"match={max_diff < 1e-4}")

    # 3. NPU bf16 run
    for dt in (torch.float16, torch.bfloat16):
        x = torch.randn(4, 32, 128, 64, dtype=dt, device="npu:0")
        y = hadamard_transform_npu(x)
        torch.npu.synchronize()
        # involution check: applying twice returns the original (H/sqrt(D) is its own inverse)
        y2 = hadamard_transform_npu(y)
        err = (y2.float() - x.float()).abs().max().item()
        print(f"[{dt}] shape={tuple(x.shape)} out.dtype={y.dtype} "
              f"involution_err(H(H(x))==x)={err:.4g}")
    print("DONE")
