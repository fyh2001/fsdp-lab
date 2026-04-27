# Issue NN: <短描述，如 "MoE all-gather hang at step 0">

> 复制本目录新建 issue：`cp -r issues/_template issues/<model>/NN-<short-slug>`

## 元信息

- **模型**：<gemma4-26b-a4b / ...>
- **首次出现**：YYYY-MM-DD
- **状态**：🔍 调查中 / ✅ 已修复 / ❌ 已绕过未根治 / ⏸️ 暂搁
- **影响**：🔴 阻塞训练 / 🟡 性能或稳定性问题 / 🟢 优化空间
- **环境**：torch=x.y.z / transformers=x.y.z / accelerate=x.y.z / 硬件=8×H100-80G

## 现象

<!--
描述清楚：
- 什么时候触发（启动时 / step 几 / 特定数据）
- 报错栈或异常 log（贴关键片段，长 log 放 assets/）
- loss / grad_norm / mem 曲线异常（截图放 assets/）
-->

## 复现

最小复现命令：

```bash
torchrun --nproc_per_node=8 repro.py
```

复现条件：
- [ ] 必须用 FSDP2（FSDP1 是否复现：）
- [ ] 必须开 grad accumulation（否则不复现）
- [ ] 与具体数据无关 / 与具体数据有关：…

## 根因

<!--
- 源码定位：哪个文件哪一行
- 调用栈
- 为什么会触发：原理解释
- 配合代码片段或 diagram
-->

## 修复 / 绕过

### 临时绕过（如有）

<!-- 配置改动 / 代码 monkey-patch -->

### 根本修复

见 `fix.patch`。已提交 upstream：
- pytorch/pytorch#xxxxx
- huggingface/transformers#xxxxx

## 相关

- 相关 issue：[../NN-xxx/](../NN-xxx/)
- upstream 讨论：<URL>
- 内部 note：[`notes/xxx.md`](../../../notes/xxx.md)

## 时间线

- YYYY-MM-DD：首次发现
- YYYY-MM-DD：定位到根因
- YYYY-MM-DD：提交 upstream PR
- YYYY-MM-DD：合入并验证
