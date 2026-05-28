
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MxfpQuantTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalElements);
  TILING_DATA_FIELD_DEF(uint64_t, nBlocks);
  TILING_DATA_FIELD_DEF(uint64_t, blocksPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, ebits);
  TILING_DATA_FIELD_DEF(uint32_t, mbits);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);  // 0=fp16, 1=bf16, 2=fp32
  TILING_DATA_FIELD_DEF(float, emax);
  TILING_DATA_FIELD_DEF(float, maxNorm);
  TILING_DATA_FIELD_DEF(float, pow2Bits);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MxfpQuant, MxfpQuantTilingData)
}
