"""Smoke + accuracy test for torch.ops.npu.mxfp_quant.

Run from outside the source dir after installing the wheel:
    cd /tmp && python /root/ops/mxfp_quant/framework/mxfp_quant_ext/test/test_binding.py
"""
import torch
import torch_npu  # noqa: F401
import mxfp_quant_ext  # registers torch.ops.npu.mxfp_quant


def _floor_log2(x):
    """Exact floor(log2(x)) for x>0 via frexp (matches kernel bit-extraction).
    frexp: x = m * 2^e, m in [0.5,1)  =>  floor(log2(x)) = e - 1."""
    _, e = torch.frexp(x)
    return (e - 1).to(x.dtype)


def ref_mxfp4(x, block_size=64):
    """Python reference matching the kernel (round-half-to-even, max_norm=6.0,
    exact exponent extraction)."""
    emax = 2.0
    max_norm = 6.0
    orig_shape = x.shape
    orig_dtype = x.dtype
    xf = x.reshape(-1, block_size).float()

    max_val = xf.abs().amax(-1, keepdim=True)
    is_zero = max_val == 0
    safe_max = torch.where(is_zero, torch.ones_like(max_val), max_val)
    log_max = _floor_log2(safe_max)
    mant = safe_max / (2.0 ** log_max)
    log_max = torch.where(mant > 1.75, log_max + 1.0, log_max)
    shared_exp = torch.clamp(log_max - emax, -127, 127)
    shared_exp = torch.where(is_zero, torch.zeros_like(shared_exp), shared_exp)
    scale = 2.0 ** shared_exp

    xs = xf / scale
    axs = xs.abs()
    pe = _floor_log2(torch.where(axs == 0, torch.ones_like(axs), axs))
    pe = torch.clamp(pe, min=0.0)
    pp = 2.0 ** pe
    tmp = xs / pp * 2.0
    rnd = torch.round(tmp)              # round half to even == CAST_RINT
    xq = rnd / 2.0 * pp
    xq = torch.clamp(xq, -max_norm, max_norm)
    xq = xq * scale
    return xq.reshape(orig_shape).to(orig_dtype)


def run(dtype, n=4096):
    x = torch.randn(n, dtype=dtype, device="npu:0") * 3.0
    y = torch.ops.npu.mxfp_quant(x, 64, 2, 3)
    torch.npu.synchronize()
    y_cpu = y.cpu().float()
    ref = ref_mxfp4(x.cpu(), 64).float()
    max_diff = (y_cpu - ref).abs().max().item()
    mism = (y_cpu != ref).sum().item()
    ok = torch.allclose(y_cpu, ref, rtol=1e-2, atol=1e-2)
    print(f"[{dtype}] shape={tuple(x.shape)} out.dtype={y.dtype} "
          f"max_abs_diff={max_diff:.4g} mismatched={mism}/{n} allclose={ok}")
    return ok


if __name__ == "__main__":
    print("ops.npu has mxfp_quant:", hasattr(torch.ops.npu, "mxfp_quant"))
    all_ok = True
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        all_ok &= run(dt)
    print("ALL PASS" if all_ok else "SOME FAILED")
