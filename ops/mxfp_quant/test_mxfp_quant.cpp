// Minimal aclnn test for MxfpQuant operator.
// Builds: g++ -I${ASCEND_HOME}/include -I${ASCEND_HOME}/opp/vendors/customize/op_api/include \
//             -L${ASCEND_HOME}/lib64 -L${ASCEND_HOME}/opp/vendors/customize/op_api/lib \
//             -lascendcl -lnnopbase -lopapi -lcust_opapi test_mxfp_quant.cpp -o test_mxfp_quant

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_mxfp_quant.h"

#define CHECK_RET(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << " line=" << __LINE__ << std::endl; return -1; } } while (0)

// fp16 helpers
static inline uint16_t Float2Half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    uint16_t h;
    if (exp >= 16) {
        h = (uint16_t)((sign << 15) | (0x1F << 10));  // inf
    } else if (exp >= -14) {
        h = (uint16_t)((sign << 15) | (((uint32_t)(exp + 15)) << 10) | (mant >> 13));
    } else if (exp >= -24) {
        uint32_t m = (mant | 0x800000) >> (-14 - exp + 13);
        h = (uint16_t)((sign << 15) | m);
    } else {
        h = (uint16_t)(sign << 15);
    }
    return h;
}
static inline float Half2Float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            int e = -14;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f = (sign << 31) | (((uint32_t)(e + 127)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | (((uint32_t)(exp - 15 + 127)) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

int CreateAclTensorFp16(const std::vector<uint16_t> &hostData,
                         const std::vector<int64_t> &shape,
                         void **deviceAddr, aclTensor **tensor) {
    size_t size = hostData.size() * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc");
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D");
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
    *tensor = aclCreateTensor(shape.data(), (int)shape.size(), ACL_FLOAT16,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), (int)shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, "aclCreateTensor");
    return 0;
}

int main() {
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit");
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice");
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream");

    // ----- Build input: shape [128] = 2 blocks of 64 -----
    const int64_t N = 128;
    std::vector<int64_t> shape = {N};
    std::vector<float> xf(N);

    // Block 0: a mix of small values
    for (int i = 0; i < 64; i++) {
        xf[i] = (i - 32) * 0.1f;  // -3.2 .. 3.1 step 0.1
    }
    // Block 1: includes some values requiring private exponent variation
    for (int i = 0; i < 64; i++) {
        xf[64 + i] = ((i % 8) - 4) * 0.7f * ((i / 8) + 1);
    }

    std::vector<uint16_t> xh(N), yh(N);
    for (int i = 0; i < N; i++) xh[i] = Float2Half(xf[i]);

    aclTensor *xT = nullptr, *yT = nullptr;
    void *xDev = nullptr, *yDev = nullptr;
    ret = CreateAclTensorFp16(xh, shape, &xDev, &xT);
    CHECK_RET(ret == 0, "create xT");
    ret = CreateAclTensorFp16(yh, shape, &yDev, &yT);
    CHECK_RET(ret == 0, "create yT");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnMxfpQuantGetWorkspaceSize(xT, /*block_size*/64, /*ebits*/2, /*mbits*/3,
                                          yT, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "GetWorkspaceSize");

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "alloc workspace");
    }
    ret = aclnnMxfpQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMxfpQuant");
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "sync");

    // Copy output back
    ret = aclrtMemcpy(yh.data(), N * sizeof(uint16_t), yDev, N * sizeof(uint16_t),
                       ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "D2H");

    // Print x | y side-by-side
    std::cout << "idx  x          y_quant" << std::endl;
    for (int i = 0; i < N; i++) {
        float y = Half2Float(yh[i]);
        std::cout << i << "\t" << xf[i] << "\t" << y << std::endl;
    }

    // Cleanup
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtFree(xDev); aclrtFree(yDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
