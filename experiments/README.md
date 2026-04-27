# Experiments

不同优化方案的**对比实验**记录。按"模型 / 优化主题"组织，结构与 [`issues/`](../issues/) 对称。

## 与其他目录的边界

| 我要记的东西的本质 | 放哪里 |
|---|---|
| 一个坏掉的现象 + 修复 | `issues/` |
| 一份原理/源码理解 | `notes/` |
| **多个方案对比 + 数据 → 选优** | **`experiments/`（本目录）** |
| 一份可直接复跑的最终配置 | `recipes/` |
| 提到 upstream 的 patch | `patches/` |

> 简单说：`experiments/` 关心**过程和对比**，`recipes/` 关心**结果和复用**。
> 一个 experiment 选出的最佳方案，通常会被固化进对应的 recipe；
> recipe 的"为什么这么配"答案，通常在对应的 experiment README 里。

## 目录结构

```text
experiments/
├── _template/                       # 实验模板，cp -r 新建实验
│   ├── README.md                    # 目标 / 矩阵 / 结果 / 结论 模板
│   └── log.md                       # 跑实验的过程实录模板
└── <model-name>/
    ├── 01-<short-slug>/
    │   ├── README.md                # 实验主文档
    │   ├── log.md                   # 跑的过程
    │   ├── configs/                 # 各 run 的配置文件
    │   ├── results/                 # 数据 / 图表 / 结论
    │   └── assets/                  # 截图、log dump 等（可选）
    └── 02-<short-slug>/
```

## 新增实验流程

1. 选好模型目录（如 `gemma4-26b-a4b/`），不存在则新建
2. 复制模板：`cp -r _template <model>/NN-<short-slug>`
3. 先填 `README.md` 的"目标"和"实验矩阵"，写清楚要回答什么问题、控制什么变量
4. 边跑边在 `log.md` 追加，每个 run 记一条
5. 结果出来后填"结果"和"结论"，画对比图放 `results/`
6. 在本 README 的索引表加一行
7. 如果产生了"最佳配置"，链回对应 [`recipes/`](../recipes/) 目录
8. 如果跑实验过程中发现了 bug，单独开一个 [`issues/`](../issues/) 项目跟踪

## 实验索引

| 模型 | 实验 | 状态 | 结论 |
|---|---|---|---|
| _暂无_ | | | |

<!--
状态：🔬 进行中 / ✅ 已结论 / ⏸️ 暂搁 / ❌ 未得出有效结论

示例：
| gemma4-26b-a4b | [01-activation-ckpt-sweep](./gemma4-26b-a4b/01-activation-ckpt-sweep/) | ✅ 已结论 | selective ckpt 比 full ckpt +26% 吞吐 |
-->
