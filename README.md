# fsdp-lab

记录在 PyTorch FSDP（FSDP1 / FSDP2）上做训练时遇到的问题、解决过程、补丁和优化。
覆盖 SFT、DPO、PPO/GRPO、继续预训练等多种训练范式。

## 仓库定位

不是教程仓库，也不是稳定的训练框架，而是一份**工程实践日志**：

- **issues/** —— 踩坑现场（按模型组织），每个问题包含复现脚本、根因分析、修复或绕过方案
- **experiments/** —— 优化方案对比实验（按模型 × 主题组织），含矩阵设计、过程实录、数据结论
- **recipes/** —— 跑通的训练配方（按训练范式 × 模型组织），可直接复用
- **notes/** —— 源码阅读笔记、原理分析等长文沉淀
- **patches/** —— 提交到 upstream（pytorch / transformers / accelerate / trl 等）的补丁汇总
- **tools/** —— 通用工具（profiler 包装、checkpoint 检查、log 解析等）

### 怎么决定写到哪里？

```text
我现在要记的内容，本质是 ——

┌─ 一个坏掉的现象 + 修复            → issues/
├─ 一份原理 / 源码理解              → notes/
├─ 多个方案对比 + 数据 → 选优       → experiments/
├─ 一份能直接复跑的最终配置         → recipes/
└─ 一个提到 upstream 的 patch       → patches/
```

`experiments/` 关心**过程和对比**，`recipes/` 关心**结果和复用**。
一个 experiment 选出的最佳方案通常会被固化进对应的 recipe；
recipe "为什么这么配"的答案，通常在对应的 experiment 里。

## 模型清单

| 模型 | 训练范式 | 状态 | Issues | Experiments | Recipe |
|---|---|---|---|---|---|
| Gemma4-26B-A4B | SFT | 🚧 进行中 | [issues/gemma4-26b-a4b](./issues/gemma4-26b-a4b/) | [experiments/gemma4-26b-a4b](./experiments/gemma4-26b-a4b/) | — |

> 状态图例：🚧 进行中 / ✅ 已完成 / ⏸️ 暂搁 / ❌ 已放弃

## 快速导航

- 想看某个模型踩过的所有坑 → [`issues/<model-name>/README.md`](./issues/)
- 想看某个模型的优化对比实验 → [`experiments/<model-name>/`](./experiments/)
- 想找一个能跑的训练脚本 → [`recipes/<task>/<model-name>/`](./recipes/)
- 想理解某个 FSDP 内部机制 → [`notes/`](./notes/)
- 想看提到 upstream 的补丁 → [`patches/`](./patches/)

## 约定

- Issue 编号统一两位数 `01/02/.../99`，按时间顺序递增
- 模型目录命名：`<架构>-<总参>-a<激活参>`（如 `gemma4-26b-a4b`、`qwen3-moe-235b-a22b`）
- 同一模型不同精度/框架变体加后缀：`gemma4-26b-a4b-fp8/`
- 每个 issue 必须包含最小复现脚本 `repro.py`（除非完全无法本地复现）

## License

MIT
