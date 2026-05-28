# sage-mxfp4-npu

昇腾 NPU 上的 **SageAttention3 风格 W4A4（MXFP4）伪量化注意力**算子与端到端 pipeline。

这个仓库包含一个自研的 AscendC 算子 `MxfpQuant`（MXFP4 块级伪量化），它的 torch_npu 绑定，以及把它 + Hadamard 旋转 + 内置 FlashAttention 串起来的「Route B」注意力 pipeline，并附带在真实 Wan2.1-T2V-1.3B 上验证画质的脚本。

> **当前状态**：伪量化（fake-quant，quant→dequant，dtype 不变）已跑通并验证画质。它用来**证明 W4A4 精度可接受**，本身不提速；真正的加速需要把伪量化算子换成真 4-bit matmul 算子（见末尾「下一步」）。

---

## 🚀 内网 FP4 机器上跑「真量化」请直接看 [`real_quant/AGENT.md`](real_quant/AGENT.md)

> 如果你在**支持原生 FP4 的昇腾机器**上（有 `torch_npu.float4_e2m1fn_x2`），想验证**真 MXFP4 量化注意力的精度与加速**，不用读下面的伪量化部分——直接：
> ```bash
> cd real_quant && bash run.sh
> ```
> `real_quant/AGENT.md` 是写给 agent 的完整手册（探测环境 → 自测精度/加速 → Wan2.1 端到端）。本仓库其余部分（`ops/`、`route_b/`）是**伪量化**实现，用于在没有 FP4 的机器上验证精度。

---

---

## 1. 这是什么 / 原理

注意力前先对 Q/K 做 **Hadamard 旋转**（抹平激活分布、减少量化离群值），再把 Q/K 量化成 **MXFP4**（每 64 个元素共享 1 个 8-bit 指数 + 每元素 4-bit：1 sign + 3 bit），注意力本身复用 CANN 内置 `npu_fusion_attention`。

- **MXFP4 网格**（每元素，乘以块共享指数 `2^shared_exp` 后）：`{0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}`
- `max_norm = 6.0`（= `2^emax × (2^(m-1)-1)/2^(m-2)`，emax=2, mbits=3）

核心算子 `MxfpQuant`：输入张量 → 按最后维 64 一组 → 求块内绝对值最大 → 算 shared exponent → 元素级 private exponent + 舍入 → 钳到 max_norm → 反缩放，输出同 shape 同 dtype。

---

## 2. 环境要求（已验证）

| 项 | 版本 |
|---|---|
| CANN | 8.2.RC2（`/usr/local/Ascend/ascend-toolkit`） |
| SoC | `ascend910_93`（910B 级，64GB HBM） |
| Python | 3.11.10 |
| torch / torch_npu | 2.5.1 / 2.5.1.post2 |
| 算子生成工具 | `msopgen`（CANN 自带） |
| 架构 | aarch64 |

> 换机器后如果 CANN 版本/SoC 不同，需要相应改 `CMakePresets.json` 里的 `ASCEND_COMPUTE_UNIT` 和 op_host 里的 `AddConfig("ascend910_93")`。

每次开新 shell 先：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
```

---

## 3. 目录结构

```
sage-mxfp4-npu/
├── README.md                     # 本文档
├── ops/mxfp_quant/               # AscendC 算子工程（msopgen 脚手架 + 自研源码）
│   ├── mxfp_quant.json           # 算子原型定义（用于 msopgen 重新生成脚手架）
│   ├── op_host/mxfp_quant.cpp     # 算子定义 + Tiling（host 侧，CPU 上算 MXFP 参数 + 切分）
│   ├── op_host/mxfp_quant_tiling.h
│   ├── op_kernel/mxfp_quant.cpp    # kernel（NPU 上跑的计算，含位操作抽指数）
│   ├── build.sh / CMakePresets.json / cmake/  # 构建链
│   ├── test_mxfp_quant.cpp         # aclnn C++ 精度自测
│   ├── bench_mxfp_quant.cpp        # aclnn C++ 性能基准
│   ├── bench_python_ref.py         # Python 参考实现的性能基准
│   └── framework/mxfp_quant_ext/   # torch_npu 绑定（torch.ops.npu.mxfp_quant）
│       ├── csrc/mxfp_quant_binding.cpp
│       ├── mxfp_quant_ext/__init__.py
│       ├── setup.py
│       └── test/test_binding.py    # Python 精度自测（vs Python 参考）
└── route_b/                       # 端到端 pipeline + Wan2.1 验证
    ├── hadamard_npu.py            # Hadamard 变换（matmul-by-Sylvester-H）
    ├── route_b_attention.py       # Hadamard + MXFP4 + npu_fusion_attention 组装
    ├── patch_wan.py               # patch 官方 Wan2.1：注入 torch_npu + SAGE_FQ 开关
    └── patch_wan2.py              # patch Wan2.1 flash_attention 的 NPU fallback
```

---

## 4. 编译 & 安装 AscendC 算子

```bash
cd ops/mxfp_quant
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
# 产物：build_out/custom_opp_hce_aarch64.run
cd build_out && ./custom_opp_hce_aarch64.run          # 安装到 opp/vendors/customize/
```

安装后算子的 aclnn 接口（`aclnnMxfpQuant` / `aclnnMxfpQuantGetWorkspaceSize`）和图算子 `MxfpQuant` 即可用。

> **如果脚手架坏了 / 换了 CANN 版本**，可以用原型 JSON 重新生成脚手架，再把 `op_host`、`op_kernel` 覆盖回去：
> ```bash
> msopgen gen -i ops/mxfp_quant/mxfp_quant.json -c ai_core-ascend910_93 -lan cpp -f pytorch -out /tmp/regen
> # 然后把本仓库的 op_host/*, op_kernel/* 拷进 /tmp/regen 覆盖，再 build.sh
> ```

### C++ 精度 / 性能自测（可选）
```bash
cd ops/mxfp_quant
g++ -std=c++17 -O2 -I$ASCEND_HOME_PATH/include \
  -I$ASCEND_HOME_PATH/opp/vendors/customize/op_api/include \
  -L$ASCEND_HOME_PATH/lib64 -L$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib \
  test_mxfp_quant.cpp -o test_mxfp_quant -lascendcl -lnnopbase -lopapi -lcust_opapi
./test_mxfp_quant        # 打印 x | y_quant，对照 MXFP4 网格
# 性能：把 test_ 换成 bench_，./bench_mxfp_quant <N> <iters>
```

---

## 5. 编译 & 安装 torch_npu 绑定

```bash
cd ops/mxfp_quant/framework/mxfp_quant_ext
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
# 自测（在源码目录外运行，避免误导入本地包）
cd /tmp && python /path/to/ops/mxfp_quant/framework/mxfp_quant_ext/test/test_binding.py
```
期望：fp16/bf16/fp32 三种 dtype 都 `allclose` 通过（参考实现用 `torch.round` 对齐 kernel 的 round-half-to-even；用 `torch.frexp` 做精确指数）。

---

## 6. Python 使用方法

### 6.1 直接调量化算子
```python
import torch, torch_npu
import mxfp_quant_ext              # 注册 torch.ops.npu.mxfp_quant
x = torch.randn(4, 64, dtype=torch.bfloat16, device="npu:0")
y = torch.ops.npu.mxfp_quant(x, 64, 2, 3)   # (x, block_size, ebits, mbits) -> 伪量化后张量
```

### 6.2 Route B 注意力（Hadamard + MXFP4 + 内置 FA）
见 `route_b/route_b_attention.py`：
```python
from route_b_attention import route_b_attention   # 需要 hadamard_npu.py、mxfp_quant_ext 在 path 上
out = route_b_attention(q, k, v)   # q,k,v: [B, N, S, D] (BNSD)，D 为 head_dim（须 2 的幂，64/128）
```

> **重要：Hadamard 作用在 head_dim（最后一维），不是 seq_len。** head_dim 在 transformer 里基本恒为 64/128（2 的幂），所以约束天然满足；seq_len 任意（5000 也没问题）。

---

## 7. 复现 Wan2.1-T2V-1.3B 画质验证

```bash
# 1. clone 官方 repo（需联网；内网请提前准备好）
cd /root && git clone https://github.com/Wan-Video/Wan2.1
# 2. 安装绑定 wheel（见第 5 节），并 pip install dashscope
# 3. 打两个 patch（注入 torch_npu + flash_attention NPU fallback + SAGE_FQ 开关）
python route_b/patch_wan.py      # 内部 REPO 路径默认 /root/Wan2.1，按需改
python route_b/patch_wan2.py
#    patch 还会修官方 RoPE 的 float64→complex128（NPU 不支持）问题：见「已知坑」
# 4. 把 route_b/hadamard_npu.py 思路内联进了 wan/modules/sage_fq.py（由 patch 生成）

cd /root/Wan2.1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

# baseline（不量化）
SAGE_FQ=0 python generate.py --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_steps 50 --ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt 'a cat walking on grass, sunny day' --save_file baseline.mp4 \
  --offload_model False --base_seed 42

# 伪量化（推荐策略：只 self-attn 的 Q/K，V 高精度，跳过首尾各 2 层）
SAGE_FQ=1 SAGE_SKIP_LAYERS=2 SAGE_NUM_LAYERS=30 \
python generate.py ... --save_file fq.mp4 ...   # 其余参数同上
```

### SAGE_FQ 环境变量
| 变量 | 默认 | 含义 |
|---|---|---|
| `SAGE_FQ` | 0 | 1=开启伪量化 |
| `SAGE_SKIP_LAYERS` | 2 | 首尾各跳过几层（保 bf16）|
| `SAGE_NUM_LAYERS` | 30 | 模型层数（1.3B=30，用于 layer 计数）|
| `SAGE_FQ_QUANT_V` | 0 | 1=连 V 也 MXFP4（默认 0：V 保高精度）|

**量化策略很关键**（否则画质崩）：只量化 self-attention 的 Q/K、V 保高精度、跳过首尾敏感层。全量化（含 cross-attn 文本 K/V + V 4-bit）会让画面细节"化掉"——这是策略问题不是算子 bug。

---

## 8. 算子设计要点 & 修过的 bug

- **位操作抽指数**：device 端没有 `log2f`/`exp2f`，kernel 用 fp32 位操作 `(bits>>23)<<23` 直接得到 `2^floor(log2|x|)`，替换了 Ln/Exp/Floor 链（最大的性能优化点）。
- **修了原型 bug 1**：`max_norm` 应为 `6.0`（原型写成 `2^emax×1.75=7.0` 错误）。
- **修了原型 bug 2**：P 量化的 mxfp4/e4m3 量纲混用（Route B 不用 P 量化，规避）。
- **修了 Reciprocal bug**：早期版本用 `Reciprocal`（Ascend vrec 是 ~2^-9 近似）算 `1/pow2_private`，把刚过 N.5 的值压到边界以下导致舍入翻车 → 改用真除法 `Div`（对 2 的幂精确）。
- **多核**：48 个 AIV core 并行。

### Wan-on-NPU 已知坑（patch 已处理）
- **RoPE complex128**：官方 `rope_apply` 用 float64→complex128，NPU 的 `aclnnCat` 不支持 → 改 float32→complex64。
- **flash_attention 断言 cuda**：官方直接调 `flash_attention`（需 CUDA flash_attn），NPU 上挂 → 加 SDPA fallback（也是 SAGE_FQ 注入点）。

---

## 9. 性能数据

### MxfpQuant 算子 vs Python 参考（干净测，device 空闲）
| N（元素） | AscendC | Python | 加速 |
|---|---|---|---|
| 64K | 0.0156 ms | 0.687 ms | **44×** |
| 512K | 0.062 ms | 0.694 ms | **11×** |
| 4M | 0.451 ms | 1.063 ms | **2.4×** |
| 16M | 1.79 ms | 2.79 ms | **1.6×** |

小张量几十倍（Python 有固定 launch 开销）；大张量收敛到 ~2×（都 memory-bound，AscendC ~35GB/s）。还有 `BlockReduceMax` 优化空间（消除每块标量往返）。

### 端到端注意力
伪量化**比 bf16 更慢**（20 步：baseline 1.84s/it vs fake-quant 2.51s/it，约 +36%），因为在 bf16 attention 之上又加了 Hadamard + 量化。**这是预期的**——伪量化只验证精度，不提速。

### 画质
官方配置（832×480 / 81 帧 / 50 步）下，推荐量化策略的伪量化视频与 bf16 baseline **肉眼几乎无差**。

---

## 10. 下一步：真 4-bit（真正的加速）

当前伪量化已证明「W4A4 精度可接受」。要拿到真实加速，把伪量化算子换成**真 4-bit matmul**：
- Q·K^T 和 P·V 的矩阵乘真的跑在 4-bit（Cube 单元），而不是 dequant 回 bf16
- 或接 CANN 的量化注意力算子（如 `aclnn_fused_infer_attention_score_v3` 的量化路径 / `swin_attention_score_quant`，需确认是否支持 MXFP4）

伪量化 → 真量化是「同一套量化数学、换底层 matmul」，精度已在本仓库验证不会崩。

---

## 给内网 AI 的最短操作清单

1. `source set_env.sh` + 设 `LD_LIBRARY_PATH`（第 2 节）
2. 建算子：`cd ops/mxfp_quant && bash build.sh && cd build_out && ./custom_opp_hce_aarch64.run`
3. 装绑定：`cd framework/mxfp_quant_ext && python setup.py bdist_wheel && pip install --force-reinstall dist/*.whl`
4. 验证：`cd /tmp && python .../test/test_binding.py`（三 dtype allclose 通过）
5. 用：`import mxfp_quant_ext; torch.ops.npu.mxfp_quant(x,64,2,3)`
6. 跑模型：参考第 7 节，注意只量化 self-attn Q/K（`SAGE_FQ=1`）
