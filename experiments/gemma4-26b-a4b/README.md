# Gemma4-26B-A4B —— Experiments

针对 Gemma4-26B-A4B 模型的优化方案对比实验。

## 实验索引

| # | 实验 | 状态 | 结论 |
|---|---|---|---|
| 01 | [activation-ckpt-sweep](./01-activation-ckpt-sweep/) | 🔬 进行中 | — |

<!--
状态：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁 / ❌ 未得出有效结论
-->

## 候选实验主题（待开展）

> 想到什么先放这里，开做时再 `cp -r ../_template NN-<slug>` 建目录。

- [x] activation checkpointing 策略 sweep（→ [01](./01-activation-ckpt-sweep/)）
- [ ] FSDP2 reshard_after_forward true/false（ZeRO-3 vs ZeRO-2 类比）
- [ ] CPU offload (params / activations) 开/关的 throughput–memory trade-off
- [ ] sequence packing 开/关 + 不同 packing 策略
- [ ] MBS × GAS × GBS 组合扫描
- [ ] Liger kernel（RMSNorm + GeGLU）开/关 + FLCE 子开关
- [ ] mixed precision: bf16 vs fp8（Hopper TE）
- [ ] torch.compile 兼容性 + 加速验证（等 PyTorch 上游修 inductor TF32 bug 后）

## 关联 recipe

- [`recipes/sft/gemma4-26b-a4b/`](../../recipes/sft/gemma4-26b-a4b/) （尚未建立）
