#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 64;  // MXFP4 default block_size

// Scalar helpers — bit-twiddle fp32 exponent field.
// Avoids depending on log2f/exp2f from libm which isn't available on device.
__aicore__ inline int FloorLog2Pos(float x) {
    // x must be > 0 and finite/normal.
    union { float f; uint32_t u; } fb;
    fb.f = x;
    return (int)((fb.u >> 23) & 0xFF) - 127;
}
__aicore__ inline float Exp2Int(int e) {
    // 2^e for integer e in [-126, 127]; saturate at edges.
    if (e > 127) e = 127;
    if (e < -126) return 0.0f;
    union { float f; uint32_t u; } fb;
    fb.u = (uint32_t)(e + 127) << 23;
    return fb.f;
}

template <typename T>
class MxfpQuantKernel {
public:
    __aicore__ inline MxfpQuantKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint64_t nBlocks,
                                 uint64_t startElemOffset,
                                 float emax, float maxNorm, float pow2Bits)
    {
        this->nBlocks_ = nBlocks;
        this->emax_ = emax;
        this->maxNorm_ = maxNorm;
        this->pow2Bits_ = pow2Bits;
        this->invPow2Bits_ = 1.0f / pow2Bits;

        uint64_t length = nBlocks * TILE_LENGTH;
        xGm_.SetGlobalBuffer((__gm__ T*)x + startElemOffset, length);
        yGm_.SetGlobalBuffer((__gm__ T*)y + startElemOffset, length);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(fp32A_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(fp32B_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(fp32C_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < nBlocks_; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t blkIdx)
    {
        LocalTensor<T> x = inQueueX_.AllocTensor<T>();
        DataCopy(x, xGm_[blkIdx * TILE_LENGTH], TILE_LENGTH);
        inQueueX_.EnQue(x);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> xIn = inQueueX_.DeQue<T>();
        LocalTensor<T> yOut = outQueueY_.AllocTensor<T>();

        LocalTensor<float> a = fp32A_.Get<float>();
        LocalTensor<float> b = fp32B_.Get<float>();
        LocalTensor<float> c = fp32C_.Get<float>();
        LocalTensor<float> rd = reduceBuf_.Get<float>();

        // 1. Cast input to fp32 (no-op copy when already fp32)
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(a, xIn, TILE_LENGTH);
        } else {
            Cast(a, xIn, RoundMode::CAST_NONE, TILE_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        // 2. abs -> reduce max -> scalar
        Abs(b, a, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        ReduceMax(rd, b, c, TILE_LENGTH);
        PipeBarrier<PIPE_V>();

        // Wait for vector before reading scalar
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float scalarMax = rd.GetValue(0);

        // 3. Compute shared exponent (scalar)
        float sharedExp = 0.0f;
        int sharedExpInt = 0;
        if (scalarMax > 0.0f) {
            int logMaxInt = FloorLog2Pos(scalarMax);
            float pow2LogMax = Exp2Int(logMaxInt);
            float mantissa = scalarMax / pow2LogMax;
            if (mantissa > 1.75f) logMaxInt += 1;
            sharedExpInt = logMaxInt - (int)emax_;
            if (sharedExpInt > 127) sharedExpInt = 127;
            if (sharedExpInt < -127) sharedExpInt = -127;
            sharedExp = (float)sharedExpInt;
        }
        float scale = Exp2Int(sharedExpInt);
        float scaleInv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        // Tell vector pipeline we wrote scalars
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        // 4. x_scaled = x * scaleInv  (a := x_scaled)
        Muls(a, a, scaleInv, TILE_LENGTH);
        PipeBarrier<PIPE_V>();

        // 5. pow2_private = 2^max(floor(log2(|x_scaled|)), 0)   (b := pow2_private)
        //    Fast path: zero the fp32 mantissa of |x_scaled| via (bits>>23)<<23.
        //    For a normal float 1.m * 2^e, clearing mantissa yields exactly 2^e
        //    = 2^floor(log2(|x|)). Replaces the Ln + Muls + Floor + Exp chain.
        Abs(b, a, TILE_LENGTH);                                 // b = |x_scaled|
        {
            LocalTensor<int32_t> bI = b.template ReinterpretCast<int32_t>();
            ShiftRight(bI, bI, (int32_t)23, TILE_LENGTH);       // logical: sign bit is 0
            PipeBarrier<PIPE_V>();
            ShiftLeft(bI, bI, (int32_t)23, TILE_LENGTH);        // b = 2^floor(log2|x_scaled|)
            PipeBarrier<PIPE_V>();
        }
        Maxs(b, b, 1.0f, TILE_LENGTH);                          // clamp private_exp>=0 (mxfp4)
        PipeBarrier<PIPE_V>();

        // 6. tmp = x_scaled / pow2_private * pow2_bits   (a := tmp)
        // Use true Div (exact for power-of-2 divisor). NOTE: do NOT use
        // Reciprocal here — Ascend vrec is ~2^-9 approximate and pushes
        // values just above an N.5 boundary below it, flipping the round.
        Div(a, a, b, TILE_LENGTH);            // a = x_scaled / pow2_private
        PipeBarrier<PIPE_V>();
        Muls(a, a, pow2Bits_, TILE_LENGTH);   // a = ... * pow2_bits
        PipeBarrier<PIPE_V>();

        // 7. Round to nearest (CAST_RINT = round half to even)
        {
            LocalTensor<int32_t> tmpI = c.template ReinterpretCast<int32_t>();
            Cast(tmpI, a, RoundMode::CAST_RINT, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            Cast(a, tmpI, RoundMode::CAST_NONE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
        }

        // 8. x_q = rounded * inv_pow2_bits * pow2_private
        Muls(a, a, invPow2Bits_, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        Mul(a, a, b, TILE_LENGTH);            // b still holds pow2_private
        PipeBarrier<PIPE_V>();

        // 10. Clamp to [-max_norm, max_norm]
        Mins(a, a, maxNorm_, TILE_LENGTH);
        PipeBarrier<PIPE_V>();
        Maxs(a, a, -maxNorm_, TILE_LENGTH);
        PipeBarrier<PIPE_V>();

        // 11. Rescale by 2^shared_exp
        Muls(a, a, scale, TILE_LENGTH);
        PipeBarrier<PIPE_V>();

        // 12. Cast back to output dtype (no-op copy when fp32)
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(yOut, a, TILE_LENGTH);
        } else {
            Cast(yOut, a, RoundMode::CAST_RINT, TILE_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        inQueueX_.FreeTensor(xIn);
        outQueueY_.EnQue(yOut);
    }

    __aicore__ inline void CopyOut(uint64_t blkIdx)
    {
        LocalTensor<T> y = outQueueY_.DeQue<T>();
        DataCopy(yGm_[blkIdx * TILE_LENGTH], y, TILE_LENGTH);
        outQueueY_.FreeTensor(y);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> fp32A_, fp32B_, fp32C_, reduceBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint64_t nBlocks_;
    float emax_;
    float maxNorm_;
    float pow2Bits_;
    float invPow2Bits_;
};


extern "C" __global__ __aicore__ void mxfp_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    uint64_t coreId = GetBlockIdx();
    uint64_t totalBlocks = tilingData.nBlocks;
    uint64_t blocksPerCore = tilingData.blocksPerCore;
    uint64_t startBlk = coreId * blocksPerCore;
    if (startBlk >= totalBlocks) return;
    uint64_t endBlk = startBlk + blocksPerCore;
    if (endBlk > totalBlocks) endBlk = totalBlocks;
    uint64_t myBlocks = endBlk - startBlk;
    uint64_t startOffset = startBlk * TILE_LENGTH;

    // Dispatch by input dtype (0=fp16, 1=bf16, 2=fp32), set by host tiling.
    if (tilingData.dtype == 2) {
        MxfpQuantKernel<float> op;
        op.Init(x, y, myBlocks, startOffset,
                tilingData.emax, tilingData.maxNorm, tilingData.pow2Bits);
        op.Process();
    } else if (tilingData.dtype == 1) {
        MxfpQuantKernel<bfloat16_t> op;
        op.Init(x, y, myBlocks, startOffset,
                tilingData.emax, tilingData.maxNorm, tilingData.pow2Bits);
        op.Process();
    } else {
        MxfpQuantKernel<half> op;
        op.Init(x, y, myBlocks, startOffset,
                tilingData.emax, tilingData.maxNorm, tilingData.pow2Bits);
        op.Process();
    }
}
