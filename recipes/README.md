# Recipes

跑通的训练配方。按**训练范式 × 模型**组织。

## 目录结构

```text
recipes/
├── sft/<model-name>/
├── dpo/<model-name>/
├── ppo/<model-name>/
└── pretrain/<model-name>/
```

每个 recipe 目录应包含：

```text
recipes/<task>/<model-name>/
├── README.md          # 硬件需求、预期吞吐/显存、最终指标、已知坑
├── train.py           # 或指向上游训练脚本的 wrapper
├── config.yaml        # 超参
├── run.sh             # 启动命令（含 torchrun 参数）
└── results/           # loss/grad 曲线截图、最终评测分数（可选）
```

## Recipe 清单

| 范式 | 模型 | 状态 | 链接 |
|---|---|---|---|
| _暂无_ | | | |

## 与 issues/ 的对应关系

每个 recipe 的 README 应在末尾列出"该模型踩过的相关坑"，链回对应的
[`issues/<model-name>/`](../issues/) 目录，方便复用者预知风险。
