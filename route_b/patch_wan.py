"""Patch the official Wan2.1 repo (on NPU box) to:
1. import torch_npu in generate.py (run CUDA-oriented code on NPU)
2. add wan/modules/sage_fq.py (Hadamard + MXFP4 fake-quant helper)
3. toggle fake-quant inside wan/modules/attention.py via env SAGE_FQ=1
Idempotent.
"""
import os
import re

REPO = "/root/Wan2.1"

# ---- 1. generate.py: inject torch_npu at top ----
gen = os.path.join(REPO, "generate.py")
src = open(gen).read()
if "import torch_npu" not in src:
    inject = ("import torch_npu  # NPU\n"
              "from torch_npu.contrib import transfer_to_npu  # NPU\n")
    src = inject + src
    open(gen, "w").write(src)
    print("generate.py: torch_npu injected")
else:
    print("generate.py: already injected")

# ---- 2. sage_fq.py ----
sage = os.path.join(REPO, "wan/modules/sage_fq.py")
open(sage, "w").write('''import torch
import mxfp_quant_ext  # noqa: F401  registers torch.ops.npu.mxfp_quant

_H = {}


def _sylvester(D, device):
    key = (D, str(device))
    if key in _H:
        return _H[key]
    assert (D & (D - 1)) == 0, f"head_dim {D} must be power of 2"
    H = torch.ones((1, 1), dtype=torch.float32, device=device)
    n = 1
    while n < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        n *= 2
    _H[key] = H
    return H


def _hadamard(x):
    D = x.shape[-1]
    s = x.shape
    out = (x.reshape(-1, D).float() @ _sylvester(D, x.device)) / (D ** 0.5)
    return out.reshape(s).to(x.dtype)


def _mxfp4(x):
    return torch.ops.npu.mxfp_quant(x.contiguous(), 64, 2, 3)


def sage_fake_quant_qkv(q, k, v):
    """q,k,v: [B, L, N, D]. Hadamard(q,k) on head_dim + MXFP4 fake-quant all."""
    return _mxfp4(_hadamard(q)), _mxfp4(_hadamard(k)), _mxfp4(v)
''')
print("sage_fq.py written")

# ---- 3. attention.py: toggle at top of attention() ----
att = os.path.join(REPO, "wan/modules/attention.py")
asrc = open(att).read()
if "SAGE_FQ" not in asrc:
    # insert right before the dispatcher's first 'if FLASH_ATTN_2_AVAILABLE'
    marker = "    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:\n        return flash_attention("
    patch = (
        "    import os as _os\n"
        "    if _os.environ.get('SAGE_FQ') == '1':\n"
        "        from .sage_fq import sage_fake_quant_qkv\n"
        "        q, k, v = sage_fake_quant_qkv(q, k, v)\n"
        "    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:\n"
        "        return flash_attention(")
    assert marker in asrc, "dispatcher marker not found"
    asrc = asrc.replace(marker, patch, 1)
    open(att, "w").write(asrc)
    print("attention.py: SAGE_FQ toggle inserted")
else:
    print("attention.py: already patched")

print("PATCH DONE")
