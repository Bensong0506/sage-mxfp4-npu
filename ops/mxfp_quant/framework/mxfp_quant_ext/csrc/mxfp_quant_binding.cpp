// torch_npu binding for the custom MxfpQuant operator.
// Pattern: direct aclnn call (the op is installed as a CANN custom opp,
// exposing aclnnMxfpQuantGetWorkspaceSize / aclnnMxfpQuant via libcust_opapi.so).
// Registers torch.ops.npu.mxfp_quant.

#include <vector>
#include <torch/library.h>
#include <torch/torch.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "aclnn_mxfp_quant.h"

namespace {

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclnnStatus _st = (expr);                                                  \
    TORCH_CHECK(_st == ACL_SUCCESS, #expr " failed, status=", _st);            \
  } while (0)

aclDataType MapDtype(at::ScalarType st) {
  switch (st) {
    case at::kHalf:     return ACL_FLOAT16;
    case at::kBFloat16: return ACL_BF16;
    case at::kFloat:    return ACL_FLOAT;
    default:
      TORCH_CHECK(false, "MxfpQuant: unsupported dtype ", st);
  }
}

// Wrap a contiguous NPU tensor as an aclTensor (ND format).
aclTensor* MakeAclTensor(const at::Tensor& t) {
  std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return aclCreateTensor(shape.data(), shape.size(), MapDtype(t.scalar_type()),
                         strides.data(), 0, ACL_FORMAT_ND,
                         shape.data(), shape.size(),
                         t.data_ptr());
}

at::Tensor mxfp_quant(const at::Tensor& x_in, int64_t block_size,
                      int64_t ebits, int64_t mbits) {
  TORCH_CHECK(x_in.device().type() == at::DeviceType::PrivateUse1,
              "MxfpQuant: input must be on NPU");
  auto x = x_in.contiguous();
  auto y = at::empty_like(x);

  aclTensor* xT = MakeAclTensor(x);
  aclTensor* yT = MakeAclTensor(y);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK(aclnnMxfpQuantGetWorkspaceSize(
      xT, block_size, ebits, mbits, yT, &workspaceSize, &executor));

  // Workspace managed by torch so it frees automatically.
  at::Tensor workspace;
  void* wsAddr = nullptr;
  if (workspaceSize > 0) {
    workspace = at::empty({(int64_t)workspaceSize},
                          x.options().dtype(at::kByte));
    wsAddr = workspace.data_ptr();
  }

  auto stream = c10_npu::getCurrentNPUStream().stream();
  ACL_CHECK(aclnnMxfpQuant(wsAddr, workspaceSize, executor, stream));

  aclDestroyTensor(xT);
  aclDestroyTensor(yT);
  return y;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("mxfp_quant(Tensor x, int block_size, int ebits, int mbits) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("mxfp_quant", TORCH_FN(mxfp_quant));
}
