import os
import glob
import torch

# Load the compiled extension which registers torch.ops.npu.mxfp_quant.
_here = os.path.dirname(__file__)
_so = glob.glob(os.path.join(_here, "_C*.so"))
if _so:
    torch.ops.load_library(_so[0])
else:
    # Fall back to the installed extension module name.
    from . import _C  # noqa: F401


def mxfp_quant(x, block_size=64, ebits=2, mbits=3):
    """MXFP4 fake-quantize on NPU. Returns a tensor of the same shape/dtype."""
    return torch.ops.npu.mxfp_quant(x, block_size, ebits, mbits)
