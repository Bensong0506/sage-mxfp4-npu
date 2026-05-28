
#include "mxfp_quant_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MxfpQuantTilingData tiling;

    // Read input shape -> total elements
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint64_t totalElements = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalElements *= xShape->GetStorageShape().GetDim(i);
    }

    // Read attrs
    auto attrs = context->GetAttrs();
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* ebitsPtr     = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* mbitsPtr     = attrs->GetAttrPointer<int64_t>(2);

    uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    uint32_t ebits     = static_cast<uint32_t>(*ebitsPtr);
    uint32_t mbits     = static_cast<uint32_t>(*mbitsPtr);

    // Compute MXFP format params on host
    float emax;
    if (ebits == 0) {
        emax = 0.0f;
    } else {
        // emax_offset = 0 for mxfp4 / mxfp8_e4m3 / mxint*
        emax = static_cast<float>((1u << (ebits - 1)));
    }
    float maxNorm;
    // matches python's `2 ** emax * (2 ** (mbits - 1) - 1) / (2 ** (mbits - 2))`
    // for mxfp4 (ebits=2, mbits=3): 4 * 3 / 2 = 6.0
    maxNorm = std::pow(2.0f, emax) * (std::pow(2.0f, (int)mbits - 1) - 1) / std::pow(2.0f, (int)mbits - 2);
    float pow2Bits = std::pow(2.0f, (int)mbits - 2);  // bits_ = mbits - 2 in python

    // Map input dtype -> kernel dispatch code (0=fp16, 1=bf16, 2=fp32)
    auto inDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeCode = 0;
    if (inDtype == ge::DT_BF16) dtypeCode = 1;
    else if (inDtype == ge::DT_FLOAT) dtypeCode = 2;

    uint64_t nBlocks = (totalElements + blockSize - 1) / blockSize;

    // Multi-core: spread blocks across AIV cores. ascend910_93 has 48 AIV.
    const uint64_t MAX_CORES = 48;
    uint64_t coreNum = (nBlocks < MAX_CORES) ? nBlocks : MAX_CORES;
    if (coreNum == 0) coreNum = 1;
    uint64_t blocksPerCore = (nBlocks + coreNum - 1) / coreNum;

    tiling.set_totalElements(totalElements);
    tiling.set_nBlocks(nBlocks);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_blockSize(blockSize);
    tiling.set_ebits(ebits);
    tiling.set_mbits(mbits);
    tiling.set_dtype(dtypeCode);
    tiling.set_emax(emax);
    tiling.set_maxNorm(maxNorm);
    tiling.set_pow2Bits(pow2Bits);

    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MxfpQuant : public OpDef {
public:
    explicit MxfpQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("block_size").AttrType(OPTIONAL).Int(64);
        this->Attr("ebits").AttrType(OPTIONAL).Int(2);
        this->Attr("mbits").AttrType(OPTIONAL).Int(3);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MxfpQuant);
}
