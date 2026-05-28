// Benchmark aclnnMxfpQuant on a realistic shape.
// Input: [B=1, H=32, Sq=2048, D=64] = 4,194,304 fp16 elements = 8 MiB
// Measures wall-clock per call after warmup.

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_mxfp_quant.h"

#define CHECK_RET(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << " line=" << __LINE__ << std::endl; return -1; } } while (0)

int main(int argc, char** argv) {
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit");
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice");
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream");

    // Shape from CLI or default
    int64_t N = (argc > 1) ? std::stoll(argv[1]) : (1LL * 32 * 2048 * 64);  // 4,194,304
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;
    int warmup = 10;

    std::vector<int64_t> shape = {N};
    size_t bytes = N * sizeof(uint16_t);

    // Build random-ish input on host (don't care about content for timing)
    std::vector<uint16_t> xh(N), yh(N);
    for (int64_t i = 0; i < N; i++) {
        xh[i] = (uint16_t)(i & 0xFFFF);  // arbitrary fp16 bit patterns
    }

    void *xDev, *yDev;
    ret = aclrtMalloc(&xDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc x");
    ret = aclrtMalloc(&yDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc y");
    ret = aclrtMemcpy(xDev, bytes, xh.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "H2D");

    std::vector<int64_t> strides = {1};
    aclTensor *xT = aclCreateTensor(shape.data(), 1, ACL_FLOAT16, strides.data(), 0,
                                     ACL_FORMAT_ND, shape.data(), 1, xDev);
    aclTensor *yT = aclCreateTensor(shape.data(), 1, ACL_FLOAT16, strides.data(), 0,
                                     ACL_FORMAT_ND, shape.data(), 1, yDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnMxfpQuantGetWorkspaceSize(xT, 64, 2, 3, yT, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "GetWorkspaceSize");

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "alloc workspace");
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ret = aclnnMxfpQuantGetWorkspaceSize(xT, 64, 2, 3, yT, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, "warmup GetWS");
        ret = aclnnMxfpQuant(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, "warmup run");
    }
    aclrtSynchronizeStream(stream);

    // Timed loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ret = aclnnMxfpQuantGetWorkspaceSize(xT, 64, 2, 3, yT, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, "GetWS");
        ret = aclnnMxfpQuant(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, "run");
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_ms = total_ms / iters;
    double gb_in  = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    double gb_out = gb_in;
    double bw_gbps = (gb_in + gb_out) / (per_ms / 1000.0);

    std::cout << "N=" << N << " iters=" << iters
              << " avg=" << per_ms << " ms"
              << " throughput=" << (N / (per_ms / 1000.0) / 1e9) << " Gelem/s"
              << " mem_bw=" << bw_gbps << " GB/s (in+out)" << std::endl;

    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtFree(xDev); aclrtFree(yDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
