"""Benchmark the Python reference vectorized_mxfp4_quantize on NPU.

Same shape as bench_mxfp_quant.cpp (default [1, 32, 2048, 64] = 4,194,304 elements).
"""
import time
import torch
import torch_npu  # noqa: F401  -- enables NPU backend


def vectorized_mxfp4_quantize(x, block_size=64):
    """MXFP4 fake-quantize matching the user's prototype.
    ebits=2, mbits=3, emax=2, max_norm=6.0 (the corrected value)."""
    ebits, mbits = 2, 3
    emax = 2.0
    max_norm = 6.0  # corrected from 7.0

    orig_shape = x.shape
    orig_dtype = x.dtype
    n_elements = x.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    if n_elements % block_size != 0:
        pad = n_blocks * block_size - n_elements
        x = torch.nn.functional.pad(x.flatten(), (0, pad)).reshape(-1, block_size)
    else:
        x = x.reshape(-1, block_size)

    x = x.to(torch.float32)

    max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
    is_zero = max_val == 0.0
    safe_max = torch.where(is_zero, torch.tensor(1.0, device=x.device), max_val)
    log_max = torch.floor(torch.log2(safe_max))
    tmp_mantissa = safe_max / (2.0 ** log_max)
    log_max = torch.where(tmp_mantissa > 1.75, log_max + 1.0, log_max)
    shared_exp = log_max - emax
    shared_exp = torch.clamp(shared_exp, min=-127, max=127)
    shared_exp = torch.where(is_zero, torch.tensor(0.0, device=x.device), shared_exp)

    x_scaled = x / (2.0 ** shared_exp)

    is_zero_e = x_scaled == 0.0
    safe_abs = torch.where(is_zero_e, torch.tensor(1.0, device=x.device), torch.abs(x_scaled))
    private_exp = torch.floor(torch.log2(safe_abs))
    private_exp = torch.where(is_zero_e, torch.tensor(0.0, device=x.device), private_exp)
    private_exp = torch.clamp(private_exp, min=0.0)

    pow2_bits = 2.0
    pow2_private = torch.exp2(private_exp)
    xq = x_scaled / pow2_private * pow2_bits
    xq = torch.sign(xq) * torch.floor(torch.abs(xq) + 0.5)
    xq = xq / pow2_bits * pow2_private
    xq = torch.clamp(xq, min=-max_norm, max=max_norm)
    xq = xq * (2.0 ** shared_exp)

    return xq.to(orig_dtype).reshape(orig_shape)


def bench(N=1 * 32 * 2048 * 64, iters=100, warmup=10):
    device = "npu:0"
    x = torch.randn(N, dtype=torch.float16, device=device).contiguous()

    for _ in range(warmup):
        _ = vectorized_mxfp4_quantize(x, 64)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = vectorized_mxfp4_quantize(x, 64)
    torch.npu.synchronize()
    t1 = time.perf_counter()

    per_ms = (t1 - t0) * 1000.0 / iters
    bytes_in = N * 2
    gb_io = (bytes_in + bytes_in) / (1024 ** 3)
    print(f"[python ref] N={N} iters={iters} "
          f"avg={per_ms:.3f} ms "
          f"throughput={N / (per_ms / 1000.0) / 1e9:.3f} Gelem/s "
          f"mem_bw={gb_io / (per_ms / 1000.0):.2f} GB/s (in+out)")


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1 * 32 * 2048 * 64
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    bench(N, iters)
