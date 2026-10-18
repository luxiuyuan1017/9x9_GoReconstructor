"""9x9 热图输出：把 loss 表画成图片，便于人工核对。

行从上往下、列从左往右；每格标注 SGF 坐标。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np

# 标题里会出现中文文件名，需要显式选用带 CJK 字形的字体，否则中文会被吞掉
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from .board import BLACK, WHITE, Board, point_to_sgf

SGF_ALPHABET = "abcdefghij"


def _shorten(text: str, limit: int) -> str:
    """标题过长时截断，避免被画布裁切。"""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def draw_heatmap(grid: Sequence[Sequence[float]], out_path: str | Path, *,
                 title: str = "", board: Optional[Board] = None,
                 actual_move: Optional[tuple[int, int]] = None,
                 colour: Optional[str] = None,
                 cmap: str = "inferno",
                 annotate: bool = True,
                 show_title: bool = False,
                 cbar_label: str = "loss") -> Path:
    """绘制 9x9 热图。

    ``grid[row][col]``；``math.inf`` 表示非法点，用浅灰色单独呈现。
    ``board`` 用于叠画棋子，``actual_move`` 用于标出被评分的那一手。
    默认不画标题（需要时传 ``show_title=True``）。
    """
    size = len(grid)
    if size == 0 or any(len(row) != size for row in grid):
        raise ValueError("grid 必须是边长一致的方阵")
    title = _shorten(title, 46)

    data = np.array(grid, dtype=float)
    illegal = ~np.isfinite(data)
    finite = data[~illegal]
    if finite.size:
        vmax = float(np.max(finite))
        vmin = float(np.min(finite))
    else:  # pragma: no cover - 防御性
        vmin, vmax = 0.0, 1.0
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    display = np.where(illegal, vmax, data)
    masked = np.ma.masked_where(illegal, display)

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")
    image = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(list(SGF_ALPHABET[:size]))
    ax.set_yticklabels(list(SGF_ALPHABET[:size]))
    ax.set_xlabel("SGF col")
    ax.set_ylabel("SGF row")
    if show_title and title:
        ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#666666", linewidth=0.5, alpha=0.4)
    ax.tick_params(which="minor", length=0)

    if annotate:
        span = vmax - vmin
        for row in range(size):
            for col in range(size):
                if illegal[row][col]:
                    ax.text(col, row, "×", ha="center", va="center",
                            color="#777777", fontsize=9)
                    continue
                value = data[row][col]
                shade = (value - vmin) / span if span else 0.0
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if shade < 0.55 else "black")

    if board is not None:
        radius = 0.32
        for row in range(size):
            for col in range(size):
                colour_at = board.get(row, col)
                if colour_at is None:
                    continue
                face = "#1a1a1a" if colour_at == BLACK else "#f5f5f5"
                ax.add_patch(plt.Circle((col, row), radius, facecolor=face,
                                        edgecolor="#555555", linewidth=0.8, zorder=3))

    if actual_move is not None:
        row, col = actual_move
        ax.add_patch(plt.Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, fill=False,
                                   edgecolor="#00e5ff", linewidth=2.2, zorder=4))
        label = point_to_sgf(row, col) + (f" {colour.upper()}" if colour else "")
        ax.text(col, row - 0.42, label, ha="center", va="bottom", fontsize=7,
                color="#00e5ff", zorder=5)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def grid_as_text(grid: Sequence[Sequence[float]], *, fmt: str = "{:6.3f}") -> str:
    """把 loss 表转成便于终端查看的文本。"""
    lines = ["     " + " ".join(f"{c:>6}" for c in SGF_ALPHABET[:len(grid)])]
    for row, values in enumerate(grid):
        cells = []
        for v in values:
            cells.append("   inf" if not math.isfinite(v) else fmt.format(v))
        lines.append(f"{SGF_ALPHABET[row]:>4} " + " ".join(cells))
    return "\n".join(lines)
