# Experiment 01: Activation Checkpointing Sweep

## 元信息

- **模型**：Gemma4-26B-A4B-it（VLM，文本-only SFT）
- **日期**：YYYY-MM-DD（开跑时填）
- **状态**：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁
- **硬件**：8×H100-80GB（NVLink）
- **框架**：FSDP2 + ms-swift / transformers

## 目标

> 在 Gemma4-26B-A4B-it + FSDP2 + 8×H100 的固定栈下，
> 比较不同 activation checkpointing（AC）策略对
> **steady step time、peak memory、tokens/s/GPU** 的影响，
> 选出该模型最佳的 AC 配置作为 recipe 默认。

预期假设（开工前的猜测，实验结束后回看是否命中）：

- [ ] AC=on 是显存最低但吞吐最差的
- [ ] AC=off (native) 显存够用且吞吐最高
- [ ] AC=offload (CPU) 通常是 throughput 最差的折中
- [ ] selective AC 在两端之间，吞吐 / 显存可调

## 实验矩阵

**控制变量**（在 P0g train-align 上锁定，不再变动）：

| 项 | 值 |
|---|---|
| seq_len (max) | 16384 |
| MBS | 1 |
| GAS | 1 |
| GBS (real unique) | 4 |
| SP (Ulysses) | 2 |
| Sharding | `full_shard` (FSDP2 native) |
| Wrap policy | `Gemma4TextDecoderLayer`, `Gemma4VisionEncoderLayer` |
| SDPA backend | `mem_efficient` only（`flash`/`math` 关）|
| Truncation | right |
| Optimizer | AdamW (lr 锁定，与 DS prod align) |
| Patches | 5 件套 sitecustomize 全部生效 |

**变化变量**：

| Run | activation_checkpointing | reshard_after_forward | offload_params | 备注 |
|---|---|---|---|---|
| `run-01-ac-on`        | `true`           | `true`  | `false` | baseline，AC=on |
| `run-02-ac-off-native`| `false`          | `true`  | `false` | 关掉 AC，native |
| `run-03-ac-offload`   | `true (offload)` | `true`  | `false` | activations CPU offload |
| `run-04-ac-selective` | `selective`      | `true`  | `false` | 仅 attn / 仅 MoE 层（policy 见 configs/） |

> 配置文件放在 `configs/`，每个 run 一份 `*.yaml` 或 `*.sh`。

## 结果

> 每个 run 跑 ≥ 30 step bench，取 step 10–29 的均值；OOM run 标记 OOM 即可。

| Run | step time (ms) | peak mem (GiB) | tokens/s/GPU (real) | full-epoch wall (min) | 备注 |
|---|---:|---:|---:|---:|---|
| run-01-ac-on         | | | | | |
| run-02-ac-off-native | | | | | |
| run-03-ac-offload    | | | | | |
| run-04-ac-selective  | | | | | |

详细图表见 [`results/`](./results/)。

## 结论 & 推荐

> 跑完后填写：

- 最佳 run：…（基于 step time / peak mem 综合）
- 选它的理由：…
- 反直觉的发现：…
- 是否已固化到 [`recipes/sft/gemma4-26b-a4b/`](../../../recipes/sft/gemma4-26b-a4b/)：是 / 否
- 适用范围 / 局限：…（例如：只在 `mem_efficient` SDPA 下成立，flash backend 下结论可能不同）

## 副产物

> 实验过程中衍生的内容：

- 发现的 bug → `issues/gemma4-26b-a4b/NN-xxx/`
- 想沉淀的原理 → `notes/xxx.md`
- 提的 upstream patch → `patches/`

（暂无）

## 相关

- 控制变量来源：基于 P0g train-align 锁定的 12 项参数
- 上一步实验：—（本实验为系列开端）
- 对应 recipe：[`recipes/sft/gemma4-26b-a4b/`](../../../recipes/sft/gemma4-26b-a4b/)（暂未建立）
- 参考：FSDP2 `apply_activation_checkpointing` 文档、Gemma4 modeling 实现
