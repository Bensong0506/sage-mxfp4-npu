"""Route B: SageAttention3-style W4A4 attention on Ascend NPU.

Pipeline:  Hadamard(Q,K) -> MXFP4 fake-quant(Q,K,V) -> npu_fusion_attention
P (softmax probs) stays high precision (built-in FA is a black box), so this
is the "W4A4 on Q/K/V" approximation of SageAttention3.

Hadamard:    matmul by cached Sylvester H_D (route_b/hadamard_npu.py)
MXFP4 quant: custom AscendC op torch.ops.npu.mxfp_quant (mxfp_quant_ext)
Attention:   torch_npu.npu_fusion_attention (built-in FlashAttention)
"""
import math
import torch
import torch_npu  # noqa: F401
import mxfp_quant_ext  # noqa: F401  registers torch.ops.npu.mxfp_quant
from hadamard_npu import hadamard_transform_npu


def mxfp4(x):
    return torch.ops.npu.mxfp_quant(x, 64, 2, 3)


def route_b_attention(q, k, v, scale=None):
    """q,k,v: [B, N, S, D] (BNSD). Returns attention output [B, N, S, D]."""
    B, N, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    qh = hadamard_transform_npu(q)
    kh = hadamard_transform_npu(k)
    qq = mxfp4(qh).contiguous()
    kq = mxfp4(kh).contiguous()
    vq = mxfp4(v).contiguous()
    out = torch_npu.npu_fusion_attention(
        qq, kq, vq, N, "BNSD", scale=float(scale))[0]
    return out


# ---------- references ----------
def _attn(q, k, v, scale):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return (p @ v.float())


def ref_fp(q, k, v, scale):
    """Full-precision attention (ground truth)."""
    return _attn(q, k, v, scale).to(q.dtype)


def ref_qkv_quant(q, k, v, scale):
    """Manual attention with Hadamard+MXFP4 on Q/K/V, P not quantized.
    This is what Route B approximates — used to validate pipeline assembly."""
    qq = mxfp4(hadamard_transform_npu(q))
    kq = mxfp4(hadamard_transform_npu(k))
    vq = mxfp4(v)
    return _attn(qq, kq, vq, scale).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, S, D = 1, 8, 512, 64
    for dt in (torch.float16, torch.bfloat16):
        q = torch.randn(B, N, S, D, dtype=dt, device="npu:0")
        k = torch.randn(B, N, S, D, dtype=dt, device="npu:0")
        v = torch.randn(B, N, S, D, dtype=dt, device="npu:0")
        scale = 1.0 / math.sqrt(D)

        out = route_b_attention(q, k, v, scale)
        torch.npu.synchronize()

        truth = ref_fp(q, k, v, scale).float()
        py = ref_qkv_quant(q, k, v, scale).float()
        o = out.float()

        def rel(a, b):
            return ((a - b).norm() / b.norm()).item()

        print(f"[{dt}] shape={tuple(q.shape)}")
        print(f"    routeB vs python-QKV-quant  rel_err={rel(o, py):.4f}  "
              f"(validates pipeline assembly; want small)")
        print(f"    routeB vs full-precision     rel_err={rel(o, truth):.4f}  "
              f"(W4A4 quantization error)")
        print(f"    python-QKV-quant vs full-prec rel_err={rel(py, truth):.4f} "
              f"(reference quant error)")
    print("DONE")
