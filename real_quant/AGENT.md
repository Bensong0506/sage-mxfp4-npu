# AGENT.md — 内网 FP4 机器上跑「真 MXFP4 量化注意力」的操作手册

> 这份文档是写给**在内网 FP4 机器上操作的 AI agent / 工程师**看的。照着做即可，无需读其他文件。
> 目标：在支持原生 FP4 的昇腾机器上，验证 **W4A4（MXFP4）真量化注意力**的精度与加速，并可选地在 Wan2.1 视频生成上端到端验证。

---

## 0. 这是什么 / 背景

我们在另一台**不支持 FP4** 的机器上已经用**伪量化**（quant→dequant，dtype 不变）证明了 MXFP4 注意力**精度可接受**（视频质量与 bf16 baseline 基本无差）。但伪量化不提速。

**真正的加速需要真 4-bit 算子**，而真 4-bit 只有**这台 FP4 机器的 torch_npu 构建**才有（原生 `float4_e2m1fn_x2` + `npu_dynamic_block_quant` + `npu_fused_infer_attention_score_v2`）。

`real_quant/` 这套代码就是把注意力切到真量化：**Hadamard 旋转 → block 量化 → 原生融合量化注意力 kernel**。写法严格对齐 Huawei 官方 MindIE-SD 的 `FP8RotateQuantFA`（已验证的 FP8 路径），只是把 dtype 参数化成 FP8 / FP4。

> ⚠️ 这套真量化代码**没有在 FP4 机器上跑过验证**（开发机没有 FP4）。所以请按下面步骤先 probe + selftest，**遇到报错按「排障」一节调**——大概率是 block_size / quant_mode / dtype 这几个参数要对齐你这台的 torch_npu 版本。

---

## 1. 环境要求

- 昇腾 NPU，CANN（建议 8.2.RC2+），且 torch_npu 构建**带 FP4**。怎么确认见第 2 步 probe。
- Python 3.11，torch 2.5.1 / torch_npu（FP4 版）。
- 每个新 shell 先：
  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

---

## 2. 第一步：探测环境（必做）

```bash
cd real_quant
python probe_env.py
```
它会打印一个能力矩阵 + 实际试跑 FP8 / FP4 注意力，最后给一行 `RECOMMENDED: fp4`（或 fp8 / bf16）。

**判读：**
- `npu_dynamic_block_quant`、`npu_fused_infer_attention_score_v2`、`float4_e2m1fn_x2` 都 `OK` 且 `PASS mode=fp4` → 这台支持真 FP4，继续。
- 只有 FP8 OK → 先用 FP8 跑通（FP4 大概率只是参数/版本问题，见排障）。
- 都 NO → 这台 torch_npu 不带这些 op，**确认是不是装错了构建**（必须是带 FP4 的内部版）。

---

## 3. 第二步：自测（精度 + 加速）

```bash
python selftest.py                      # 自动测所有能跑的模式
# 或指定 shape（B N S D，BNSD）和模式：
python selftest.py --mode fp4 --shape 1 24 4096 128 --iters 50
```
期望输出（每个模式一行）：
```
[bf16 npu_fusion_attention] X.XXX ms (speed baseline)
[fp4] rel_err_vs_fp=0.xx  latency=Y.YYY ms  speedup_vs_bf16=Z.ZZx
```
**成功标准：**
- `rel_err_vs_fp` 不大（随机数据 0.1~0.25 属正常，真实激活会更低）。
- `speedup_vs_bf16 > 1`（这才是真加速；若 <1 说明没真的走 4-bit kernel，见排障）。

一键跑 probe+selftest：`bash run.sh`（或 `bash run.sh fp4`）。

---

## 4. 第三步（可选）：Wan2.1 视频端到端验证

```bash
# 准备：模型权重路径（原始 Wan 格式：diffusion_pytorch_model.safetensors + VAE + T5 + google/）
# clone 官方 repo
git clone https://github.com/Wan-Video/Wan2.1 /root/Wan2.1
pip install dashscope                      # generate.py 的可选依赖
# 打 patch（注入 torch_npu + 修 RoPE + 接真量化 hook）
python patch_wan_realquant.py --wan /root/Wan2.1 --realquant "$(pwd)"

cd /root/Wan2.1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# baseline（bf16）
python generate.py --task t2v-1.3B --size '832*480' --frame_num 81 --sample_steps 50 \
  --ckpt_dir /path/to/Wan2.1-T2V-1.3B --prompt 'a cat walking on grass, sunny day' \
  --save_file baseline.mp4 --offload_model False --base_seed 42

# 真 FP4 量化（只 self-attn，跳过首尾各 2 层）
SAGE_MODE=fp4 SAGE_SKIP_LAYERS=2 SAGE_NUM_LAYERS=30 \
python generate.py ... --save_file fp4.mp4 ...   # 其余参数同上

# 对比 baseline.mp4 vs fp4.mp4：画质应基本一致；记录两者总耗时（FP4 应更快）
```
> Wan2.1-1.3B：`SAGE_NUM_LAYERS=30`；14B 用 40。`SAGE_MODE` 可选 `bf16|fp8|fp4`。

---

## 5. 排障（FP4 没跑通时按顺序试）

真量化 kernel 对参数敏感，开发机没法预先验证，遇到报错这样调：

1. **`npu_dynamic_block_quant` 不接受 float4_e2m1fn_x2 / 报 block size 错**
   → 改 `quant_attention.py` 里 `q_block/kv_block/col_block`。MX 标准是 32；FP8 用的是 128/256。
   先试 `quant_attention(..., q_block=128, kv_block=128, col_block=128)`，再试 32。
2. **`npu_fused_infer_attention_score_v2` 报 quant_mode 不支持**
   → `query_quant_mode/key_quant_mode/value_quant_mode` 当前写的 7（FP8 用的值）。FP4 可能是别的值；
   查这台 torch_npu 的 `npu_fused_infer_attention_score_v2` 文档/签名里 quant_mode 的取值表，换成 FP4 对应的。
3. **`speedup < 1`（精度对但没加速）**
   → 说明 kernel 走了 dequant 回退而不是真 4-bit。确认 dtype 真的是 `float4_e2m1fn_x2`，且 `npu_fused_infer_attention_score_v2` 确实吃量化输入（probe 的 live test 应已 PASS）。
4. **找不到 op**
   → 对照 Huawei MindIE-SD 的真实用法（这就是我们抄的来源）：
   `git clone https://gitcode.com/Ascend/MindIE-SD`，看
   `mindiesd/quantization/layer.py: FP8RotateQuantFA` 和
   `mindiesd/layers/quant/block_quant.py: fa_block_quant_preprocess`。
   把它们的 op 名 / 参数对齐到你这台的版本。
5. **想要最稳的真量化**：直接 `pip install mindiesd`（如内网有源），用它的 `FP8RotateQuantFA` 跑 FP8 先拿到加速基线，再按它的模式扩到 FP4。

---

## 6. 关键文件

| 文件 | 作用 |
|---|---|
| `probe_env.py` | 探测 FP8/FP4 能力，给推荐模式 |
| `quant_attention.py` | 真量化注意力（旋转→block_quant→fused_infer_attention_v2），FP8/FP4 可切 |
| `selftest.py` | 精度 vs bf16 + 加速 vs bf16 |
| `patch_wan_realquant.py` | 把 Wan2.1 注意力切到真量化（env `SAGE_MODE`）|
| `run.sh` | probe + selftest 一键跑 |
| `../ops/mxfp_quant/` | 我们自研的 MXFP4 **伪量化** AscendC 算子（精度参照，开发机用）|
| `../route_b/` | 伪量化 pipeline（已验证画质，开发机用）|

---

## 7. 一句话总结给 agent

> 跑 `bash real_quant/run.sh`。看到 `[fp4] ... speedup_vs_bf16=>1` 且 `rel_err` 不大，就说明真 MXFP4 注意力在这台机器上 work 且更快。报错就按第 5 节对照 MindIE-SD 调 block_size / quant_mode。然后按第 4 节跑 Wan2.1 视频做端到端确认。
