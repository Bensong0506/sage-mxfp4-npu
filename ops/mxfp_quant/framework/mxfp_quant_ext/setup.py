import os
import glob
import torch
import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

TN = os.path.dirname(torch_npu.__file__)
CANN = os.environ.get("ASCEND_HOME_PATH",
                      "/usr/local/Ascend/ascend-toolkit/latest")
CUST = os.path.join(CANN, "opp/vendors/customize/op_api")

include_dirs = [
    os.path.join(TN, "include"),
    os.path.join(TN, "include/third_party/acl/inc"),
    os.path.join(CANN, "include"),
    os.path.join(CUST, "include"),
]

library_dirs = [
    os.path.join(TN, "lib"),
    os.path.join(CANN, "lib64"),
    os.path.join(CUST, "lib"),
]

libraries = [
    "torch_npu",
    "ascendcl",
    "nnopbase",
    "opapi",
    "cust_opapi",
]

ext = CppExtension(
    name="mxfp_quant_ext._C",
    sources=glob.glob("csrc/*.cpp"),
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=["-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_link_args=["-Wl,-rpath," + os.path.join(CUST, "lib"),
                     "-Wl,-rpath," + os.path.join(CANN, "lib64")],
)

setup(
    name="mxfp_quant_ext",
    version="0.1",
    packages=["mxfp_quant_ext"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
