# Gemma4-26B-A4B —— Experiments

针对 Gemma4-26B-A4B 模型的优化方案对比实验。

## 实验索引

| # | 实验 | 状态 | 结论 |
|---|---|---|---|
| _暂无_ | | | |

<!--
状态：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁 / ❌ 未得出有效结论

示例：
| 01 | [activation-ckpt-sweep](./01-activation-ckpt-sweep/) | ✅ 已结论 | selective ckpt 比 full ckpt +26% 吞吐 |
| 02 | [fsdp1-vs-fsdp2](./02-fsdp1-vs-fsdp2/)               | 🔬 进行中 | — |
-->

## 候选实验主题（待开展）

> 想到什么先放这里，开做时再 cp 模板建目录。

- [ ] activation checkpointing 策略 sweep（none / full / selective / moe-only）
- [ ] FSDP1 vs FSDP2 吞吐对比
- [ ] CPU offload 开/关的 throughput–memory trade-off
- [ ] 不同 sequence packing 策略（concat / pad / sample-packing）
- [ ] mbs × grad_accum 组合扫描
- [ ] mixed precision: bf16 vs fp8

## 关联 recipe

- [`recipes/sft/gemma4-26b-a4b/`](../../recipes/sft/gemma4-26b-a4b/) （尚未建立）
