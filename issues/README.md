# Issues

按**模型**分组的踩坑档案。每个子目录对应一个具体模型，目录内的 `README.md` 是该模型的问题索引。

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
