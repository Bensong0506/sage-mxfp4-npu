"""Probe the NPU environment for real-quant attention capability.

Run this FIRST on the FP4 machine. It checks which torch_npu ops/dtypes exist
and actually tries a tiny FP8 / FP4 quantized-attention call, then prints a
capability matrix and the recommended mode for selftest.py / the Wan demo.

Exit code 0 always; read the printed RECOMMENDED line.
"""
import math
import torch

print("=" * 60)
print("real-quant attention environment probe")
print("=" * 60)

try:
    import torch_npu
    print(f"torch        = {torch.__version__}")
    print(f"torch_npu    = {getattr(torch_npu, '__version__', '?')}")
    print(f"npu available= {torch.npu.is_available()}")
except Exception as e:
    print("torch_npu import FAILED:", e)
    print("RECOMMENDED: none (no NPU / torch_npu)")
    raise SystemExit(0)


def has(obj, name):
    return hasattr(obj, name)


ops = {
    "npu_dynamic_block_quant": has(torch_npu, "npu_dynamic_block_quant"),
    "npu_fused_infer_attention_score_v2": has(torch_npu, "npu_fused_infer_attention_score_v2"),
    "npu_fused_infer_attention_score": has(torch_npu, "npu_fused_infer_attention_score"),
    "npu_dynamic_mx_quant": has(torch_npu, "npu_dynamic_mx_quant"),
    "npu_quant_matmul": has(torch_npu, "npu_quant_matmul"),
}
dtypes = {
    "float8_e4m3fn": has(torch, "float8_e4m3fn") or has(torch_npu, "float8_e4m3fn"),
    "float4_e2m1fn_x2": has(torch_npu, "float4_e2m1fn_x2"),
    "float8_e8m0fnu": has(torch_npu, "float8_e8m0fnu"),
}

print("\n-- ops --")
for k, v in ops.items():
    print(f"  {'OK ' if v else 'NO '} {k}")
print("-- dtypes --")
for k, v in dtypes.items():
    print(f"  {'OK ' if v else 'NO '} {k}")


def try_mode(mode):
    """Actually run a tiny quant_attention in this mode; return (ok, msg)."""
    try:
        from quant_attention import quant_attention
        q = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16, device="npu:0")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = quant_attention(q, k, v, mode=mode, rotate=True)
        torch.npu.synchronize()
        return True, f"output {tuple(out.shape)} {out.dtype}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"


print("\n-- live attention test --")
results = {}
for mode in ("fp8", "fp4"):
    ok, msg = try_mode(mode)
    results[mode] = ok
    print(f"  {'PASS' if ok else 'FAIL'} mode={mode}: {msg}")

if results.get("fp4"):
    rec = "fp4"
elif results.get("fp8"):
    rec = "fp8"
else:
    rec = "bf16 (no real-quant path available on this build)"
print("\nRECOMMENDED:", rec)
print("Next: python selftest.py --mode", rec.split()[0])
