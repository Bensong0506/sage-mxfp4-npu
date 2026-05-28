"""Self-test for real-quant attention: accuracy vs bf16 + speed vs bf16.

Usage (on the FP4 machine, after probe_env.py):
    python selftest.py                 # auto: tests every mode that runs
    python selftest.py --mode fp4      # force a mode
    python selftest.py --shape 1 24 4096 128 --iters 50

For each working mode it prints:
  - rel_err vs full-precision attention (quality)
  - latency vs bf16 fused attention (the real speedup number you care about)
"""
import argparse
import math
import time
import torch
import torch_npu  # noqa: F401

from quant_attention import quant_attention, _ref_attention


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 24, 4096, 128],
                    help="B N S D (BNSD)")
    ap.add_argument("--mode", default="auto", choices=["auto", "fp8", "fp4", "bf16"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    B, N, S, D = args.shape
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = "npu:0"
    print(f"shape BNSD={args.shape} dtype={args.dtype} iters={args.iters}")

    q = torch.randn(B, N, S, D, dtype=dt, device=dev)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # ground truth (full precision)
    truth = _ref_attention(q, k, v)

    # bf16 fused attention baseline for speed
    def bf16_fa():
        return torch_npu.npu_fusion_attention(
            q, k, v, N, "BNSD", scale=1.0 / math.sqrt(D))[0]
    t_bf16 = bench(bf16_fa, args.iters)
    print(f"[bf16 npu_fusion_attention] {t_bf16:.3f} ms (speed baseline)")

    modes = ["fp8", "fp4"] if args.mode == "auto" else [args.mode]
    for mode in modes:
        if mode == "bf16":
            continue
        try:
            out = quant_attention(q, k, v, mode=mode, rotate=True)
            torch.npu.synchronize()
        except Exception as e:
            print(f"[{mode}] SKIP ({type(e).__name__}: {str(e)[:120]})")
            continue
        err = rel_err(out, truth)
        t = bench(lambda: quant_attention(q, k, v, mode=mode, rotate=True), args.iters)
        speedup = t_bf16 / t if t > 0 else 0
        print(f"[{mode}] rel_err_vs_fp={err:.4f}  latency={t:.3f} ms  "
              f"speedup_vs_bf16={speedup:.2f}x")

    print("\nNote: rel_err on random data is worst-case; real activations + "
          "Hadamard give lower error. speedup<1 means slower (check that the "
          "real low-bit kernel is actually engaged, not falling back).")


if __name__ == "__main__":
    main()
