"""Patch an official Wan2.1 checkout to run REAL quantized attention on NPU.

Applies, idempotently:
  1. inject torch_npu into generate.py (run CUDA-oriented code on NPU)
  2. RoPE float64->float32 (NPU has no complex128)
  3. flash_attention(): NPU fallback + SAGE_MODE real-quant hook that routes
     self-attention through real_quant/quant_attention.py (fp8 / fp4 / bf16)

Usage:
    python patch_wan_realquant.py --wan /root/Wan2.1 --realquant /path/to/real_quant
Then run generate.py with env SAGE_MODE=fp4 (or fp8 / bf16).

Env vars honored at runtime:
    SAGE_MODE        bf16 | fp8 | fp4   (default unset = no quant)
    SAGE_SKIP_LAYERS first/last N layers kept bf16 (default 2)
    SAGE_NUM_LAYERS  model layer count (Wan2.1-1.3B=30, 14B=40)
"""
import argparse
import os


def patch(wan, realquant):
    # 1. generate.py torch_npu inject
    gen = os.path.join(wan, "generate.py")
    s = open(gen).read()
    if "import torch_npu" not in s:
        s = ("import torch_npu  # NPU\nfrom torch_npu.contrib import transfer_to_npu  # NPU\n") + s
        open(gen, "w").write(s)
        print("generate.py: torch_npu injected")
    else:
        print("generate.py: already injected")

    # 2. RoPE float64 -> float32
    model = os.path.join(wan, "wan/modules/model.py")
    m = open(model).read()
    n0 = m.count("torch.float64")
    m = m.replace("torch.arange(0, dim, 2).to(torch.float64)",
                  "torch.arange(0, dim, 2).to(torch.float32)")
    m = m.replace("x[i, :seq_len].to(torch.float64)",
                  "x[i, :seq_len].to(torch.float32)")
    open(model, "w").write(m)
    print(f"model.py: RoPE float64->float32 ({n0}->{m.count('torch.float64')})")

    # 3. attention.py flash_attention NPU fallback + real-quant hook
    att = os.path.join(wan, "wan/modules/attention.py")
    a = open(att).read()
    if "SAGE_REALQUANT_HOOK" not in a:
        marker = "    half_dtypes = (torch.float16, torch.bfloat16)"
        block = (
            "    # --- SAGE_REALQUANT_HOOK: NPU real-quant / SDPA fallback ---\n"
            "    import os as _os, sys as _sys, math as _math\n"
            "    import torch.nn.functional as _F\n"
            "    if (not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE)) or q.device.type != 'cuda':\n"
            "        _out_dtype = q.dtype\n"
            "        _mode = _os.environ.get('SAGE_MODE')\n"
            "        _self_attn = (q.shape[1] == k.shape[1])\n"
            f"        _RQ = {realquant!r}\n"
            "        if _mode in ('fp8', 'fp4') and _self_attn:\n"
            "            if _RQ not in _sys.path:\n"
            "                _sys.path.insert(0, _RQ)\n"
            "            from quant_attention import quant_attention as _qa\n"
            "            # optional layer skip\n"
            "            _skip = int(_os.environ.get('SAGE_SKIP_LAYERS', '2'))\n"
            "            _nl = int(_os.environ.get('SAGE_NUM_LAYERS', '0'))\n"
            "            global _SAGE_LC\n"
            "            try:\n"
            "                _SAGE_LC\n"
            "            except NameError:\n"
            "                _SAGE_LC = 0\n"
            "            _do = True\n"
            "            if _nl > 0:\n"
            "                _idx = _SAGE_LC % _nl\n"
            "                _SAGE_LC += 1\n"
            "                _do = (_skip <= _idx < _nl - _skip)\n"
            "            if _do:\n"
            "                # Wan q,k,v are [B,S,N,D]; quant_attention wants BNSD\n"
            "                _o = _qa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
            "                         mode=_mode, rotate=True)\n"
            "                return _o.transpose(1, 2).contiguous().to(_out_dtype)\n"
            "        # bf16 SDPA fallback (also covers cross-attn & skipped layers)\n"
            "        _q = q.transpose(1, 2).to(dtype); _k = k.transpose(1, 2).to(dtype); _v = v.transpose(1, 2).to(dtype)\n"
            "        if q_scale is not None:\n"
            "            _q = _q * q_scale\n"
            "        _o = _F.scaled_dot_product_attention(_q, _k, _v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)\n"
            "        return _o.transpose(1, 2).contiguous().to(_out_dtype)\n"
            "    half_dtypes = (torch.float16, torch.bfloat16)"
        )
        assert marker in a, "attention.py marker not found"
        a = a.replace(marker, block, 1)
        open(att, "w").write(a)
        print("attention.py: real-quant hook + NPU fallback inserted")
    else:
        print("attention.py: already patched")
    print("PATCH DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan", default="/root/Wan2.1")
    ap.add_argument("--realquant", required=True,
                    help="absolute path to this repo's real_quant/ dir")
    args = ap.parse_args()
    patch(args.wan, os.path.abspath(args.realquant))
