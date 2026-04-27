# Experiment NN: <短描述，如 "Activation Checkpointing Sweep">

> 复制本目录新建实验：`cp -r experiments/_template experiments/<model>/NN-<short-slug>`

## 元信息

- **模型**：<gemma4-26b-a4b / ...>
- **日期**：YYYY-MM-DD
- **状态**：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁 / ❌ 未得出有效结论
- **作者**：<可选>
- **基础环境**：torch=x.y.z / transformers=x.y.z / 硬件=8×H100-80G

## 目标

<!--
用一句话回答：这个实验要回答什么具体问题？
- 好：在 8×H100 上，不同 activation ckpt 策略对 throughput vs memory 的 trade-off？
- 差：试试 activation checkpointing
-->

## 实验矩阵

<!--
列清楚：哪些变量在变，哪些控制不变。
-->

| Run | 关键变量 | 其他配置 | 备注 |
|---|---|---|---|
| run-01 | … | … | baseline |
| run-02 | … | … | |
| run-03 | … | … | |

**控制变量**：
- seq_len = 4096
- global_batch_size = 256
- optimizer = AdamW (lr=1e-5)
- 其他 …

**变化变量**：
- 见上表

## 结果

| Run | 指标 A | 指标 B | 指标 C | 备注 |
|---|---|---|---|---|
| run-01 | | | | |
| run-02 | | | | |
| run-03 | | | | |

详细图表见 [`results/`](./results/)。

## 结论 & 推荐

<!--
- 哪一个方案最佳？理由（数据支撑）
- 是否已固化到 recipes/？
- 反直觉的发现
- 局限性 / 适用范围
-->

## 副产物

<!--
跑实验过程中衍生的内容，链到对应位置：
- 发现的 bug → issues/.../NN-xxx/
- 想深入理解的原理 → notes/xxx.md
- 提的 upstream patch → patches/
-->

- _暂无_

## 相关

- 相关实验：[../NN-xxx/](../NN-xxx/)
- 对应 recipe：[`recipes/<task>/<model>/`](../../../recipes/)
- 参考资料：<论文 / 博客 / PR 链接>
