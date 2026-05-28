"""Patch flash_attention() in Wan2.1 to fall back to SDPA on NPU (the model
calls flash_attention directly, bypassing the attention() dispatcher), and to
apply the SAGE_FQ fake-quant toggle there. Idempotent."""
REPO = "/root/Wan2.1"
att = f"{REPO}/wan/modules/attention.py"
s = open(att).read()

if "SAGE_FQ_FALLBACK" not in s:
    marker = "    half_dtypes = (torch.float16, torch.bfloat16)"
    block = (
        "    # --- NPU / no-flash-attn fallback (SAGE_FQ_FALLBACK) ---\n"
        "    import os as _os\n"
        "    import torch.nn.functional as _F\n"
        "    if (not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE)) or q.device.type != 'cuda':\n"
        "        _out_dtype = q.dtype\n"
        "        if _os.environ.get('SAGE_FQ') == '1':\n"
        "            from .sage_fq import sage_fake_quant_qkv\n"
        "            q, k, v = sage_fake_quant_qkv(q, k, v)\n"
        "        _q = q.transpose(1, 2).to(dtype)\n"
        "        _k = k.transpose(1, 2).to(dtype)\n"
        "        _v = v.transpose(1, 2).to(dtype)\n"
        "        if q_scale is not None:\n"
        "            _q = _q * q_scale\n"
        "        _o = _F.scaled_dot_product_attention(_q, _k, _v, is_causal=causal,\n"
        "                                             dropout_p=dropout_p, scale=softmax_scale)\n"
        "        return _o.transpose(1, 2).contiguous().to(_out_dtype)\n"
        "    half_dtypes = (torch.float16, torch.bfloat16)"
    )
    assert marker in s, "half_dtypes marker not found"
    s = s.replace(marker, block, 1)
    open(att, "w").write(s)
    print("flash_attention NPU fallback + SAGE_FQ inserted")
else:
    print("already patched")
print("DONE")
