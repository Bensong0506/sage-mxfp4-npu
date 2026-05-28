"""Real (not fake) quantized attention for Ascend NPU.

This mirrors the *validated* Huawei pattern from MindIE-SD's `FP8RotateQuantFA`
(mindiesd/quantization/layer.py): rotate Q/K -> block-quant Q/K/V -> call the
native fused quantized attention kernel `npu_fused_infer_attention_score_v2`.
We parameterize the quant dtype so the same path runs in FP8 (e4m3, proven) or
FP4 (e2m1, the W4A4 target — confirm on the FP4-capable build).

Unlike ../route_b (which uses *fake* quant to validate accuracy on any box),
this module produces a REAL low-bit attention and is the path to actual speedup.
It REQUIRES a torch_npu build that exposes:
    - torch_npu.npu_dynamic_block_quant
    - torch_npu.npu_fused_infer_attention_score_v2
    - torch_npu.float8_e4m3fn  (FP8)  and/or  torch_npu.float4_e2m1fn_x2 (FP4)
Run ./probe_env.py first to check.

Reference (gitcode): https://gitcode.com/Ascend/MindIE-SD
  mindiesd/quantization/layer.py  -> class FP8RotateQuantFA
  mindiesd/layers/quant/block_quant.py -> fa_block_quant_preprocess
"""
import math
import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover - only meaningful on NPU
    torch_npu = None

_H_CACHE = {}


def hadamard_matrix(d, device, dtype=torch.float32):
    """Normalized Sylvester Hadamard matrix H_d (d power of 2). (x @ H)/sqrt(d)
    is the Walsh-Hadamard transform; applying the same H to Q and K preserves
    Q·K^T exactly while smoothing the per-channel distribution for quantization."""
    key = (d, str(device), str(dtype))
    if key in _H_CACHE:
        return _H_CACHE[key]
    assert (d & (d - 1)) == 0, f"head_dim {d} must be a power of 2 for Hadamard"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    n = 1
    while n < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        n *= 2
    H = H / math.sqrt(d)
    _H_CACHE[key] = H
    return H


def _quant_dtype(mode):
    if mode == "fp8":
        return torch_npu.float8_e4m3fn
    if mode == "fp4":
        # E2M1 packed two-per-byte; only on FP4-capable torch_npu builds.
        return torch_npu.float4_e2m1fn_x2
    raise ValueError(f"unknown quant mode {mode!r} (use 'fp8' or 'fp4')")


def _ref_attention(q, k, v):
    """bf16/fp16 reference (ground truth), BNSD."""
    d = q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    p = torch.softmax(s, dim=-1)
    return (p @ v.float()).to(q.dtype)


def quant_attention(q, k, v, mode="fp8", rotate=True,
                    q_block=128, kv_block=256, col_block=128):
    """Real quantized attention.

    Args:
        q, k, v: [B, N, S, D] (BNSD), bf16/fp16. D (head_dim) must be power of 2.
        mode: 'bf16' (reference, no quant), 'fp8' (e4m3, proven), 'fp4' (e2m1, W4A4 target).
        rotate: apply Hadamard rotation to Q,K before quant (recommended for low-bit).
        q_block/kv_block/col_block: block-quant sizes (defaults match MindIE-SD FP8 FA).
    Returns:
        [B, N, S, D] attention output, same dtype as q.
    """
    assert q.dim() == 4, "expects BNSD"
    B, N, S, D = q.shape

    if rotate:
        H = hadamard_matrix(D, q.device, q.dtype)
        q = torch.matmul(q, H)
        k = torch.matmul(k, H)

    if mode == "bf16":
        return _ref_attention(q, k, v)

    assert torch_npu is not None, "torch_npu required for real quant"
    DT = _quant_dtype(mode)

    # Block-quant Q/K/V (npu_dynamic_block_quant works on BNSD with batch squeezed).
    # NOTE: MindIE-SD squeezes the batch dim (assumes B==1 per call). For B>1 loop.
    def bq(x, row_block):
        xq, xs = torch_npu.npu_dynamic_block_quant(
            x.squeeze(0), dst_type=DT,
            row_block_size=row_block, col_block_size=col_block)
        return xq.unsqueeze(0), xs.unsqueeze(0)

    if B != 1:
        outs = [quant_attention(q[i:i + 1], k[i:i + 1], v[i:i + 1], mode=mode,
                                rotate=False, q_block=q_block, kv_block=kv_block,
                                col_block=col_block) for i in range(B)]
        return torch.cat(outs, dim=0)

    qb, qs = bq(q, q_block)
    kb, ks = bq(k, kv_block)
    vb, vs = bq(v, kv_block)

    out = torch_npu.npu_fused_infer_attention_score_v2(
        qb, kb, vb,
        input_layout="BNSD",
        num_query_heads=N,
        softmax_scale=1.0 / math.sqrt(D),
        pre_tokens=2147483647,
        next_tokens=2147483647,
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=qs,
        dequant_scale_key=ks,
        dequant_scale_value=vs,
        out_dtype=q.dtype,
    )[0]

    if out.shape[2] != S:
        out = out[:, :, :S, :]
    return out
