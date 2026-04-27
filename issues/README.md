# Issues

按**模型**分组的踩坑档案。每个子目录对应一个具体模型，目录内的 `README.md` 是该模型的问题索引。

> 与 [`experiments/`](../experiments/) 的边界：
> - **issues/** = 反应式（"东西坏了 → 我要修"），单点 bug 修复
> - **experiments/** = 主动式（"东西没坏，但我要在多个方案里挑最好"），对比研究
>
> 跑实验过程中**衍生出的 bug** 单独在 `issues/` 立项，并在 experiment 的"副产物"段链回。

## 目录结构

```text
issues/
├── _template/              # 标准 issue 模板，新增问题时 cp -r 复制
├── <model-name>/
│   ├── README.md           # 该模型的问题索引 + 时间线 + 当前状态
│   ├── 01-<short-slug>/
│   ├── 02-<short-slug>/
│   └── ...
```

## 新增 issue 的标准流程

1. 如果是新模型：先建模型目录和模型级 `README.md`（参考已有模型，如 `gemma4-26b-a4b/README.md`）
2. 复制模板：`cp -r _template <model>/NN-<short-slug>`
3. 填写问题 README、提供最小复现脚本
4. 在模型级 `README.md` 的索引表里加一行
5. 修复后更新状态字段

## 模型清单

| 模型 | 状态 |
|---|---|
| [gemma4-26b-a4b](./gemma4-26b-a4b/) | 🚧 进行中 |
