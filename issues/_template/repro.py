"""Minimal reproduction script.

启动：torchrun --nproc_per_node=<N> repro.py

目标：用最少的依赖、最小的模型/数据复现问题，方便他人（包括 upstream
maintainer）快速验证。能用 dummy 数据就别用真实数据集；能用小模型暴露
问题就别用大模型。
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("TODO: fill in minimal repro")


if __name__ == "__main__":
    main()
