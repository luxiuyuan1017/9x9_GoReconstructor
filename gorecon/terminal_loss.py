"""子程序一：固定终盘点位 loss（终端 loss）。

输入目标局面，输出黑、白各一张 9x9 浮点代价表。它在整次搜索中**只计算一次**并保持不变。

固定空区定义（项目说明第 4 节）：

* 在目标终盘中用上下左右连接划分完整空区；
* 统计每个空区相邻棋子的颜色集合，棋盘边缘只充当边界，不是一种棋子颜色；
* 只接触黑棋的空区视为被黑方包围，只接触白棋的视为被白方包围；
* 同时接触双方、或完全没有接触棋子的空区，不判为任何一方完全包围。

首版代价规则（默认为待验证的实现默认值）。按当前落子颜色看终局信息：

==================================  ============================================
当前落子颜色的终局信息                固定点位代价
==================================  ============================================
终盘为同色棋子                       0（全局唯一最小值，每放对一个子就减 1 阶）
终盘为对方棋子                       一档适中的有限罚分
终盘为对方棋子，且该点在本方团块内部    更低一档（角/边/常规分开配置，本版同值）
终盘为空，空区只被对方完全包围         基础罚分 + 面积 * 面积系数
终盘为空，其余情况                    一档大的有限罚分（自己完全包围，或中立空区）
==================================  ============================================

四个要点：

* **"团块内部"的定义**：该点相邻的**全部在盘内**棋子都是同色。角上只有 2 个邻点、
  边上 3 个、其余 4 个，只要这些邻点全是本方棋子就算内部。团块内部往往是曾经发生
  提子的地方，历史上更可能真的放过对方的子，所以该点处对方的代价更低；
  角/边/常规三档分开配置，便于按位置分别调整；
* **对方包围空区不是一上来就很贵**：它由 ``基础罚分 + 面积 * 面积系数`` 决定，小面积
  时只是适中，随面积线性增长；具体曲线由这两个系数配置，不要硬编码成"很大"；
* **异色点罚分不能是最贵的一档**：终盘是对方棋子的那一点，正是搜索把棋子放对位置的
  主要路径（对方这块子被提掉、自己在此落子才可能变成同色）；
* **全部有限**：终局属于自己的子也可能在这条重建历史里被提掉后补回来，无限代价会
  剪掉合法路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .board import BLACK, WHITE, Board
from .config import TerminalLossConfig

#: 代价表所在点位的分类标签
SAME_COLOUR = "same_colour"
OPPONENT_TERRITORY = "opponent_territory"
CLOSED_EMPTY = "closed_empty"          # 自己完全包围的空区
NEUTRAL_EMPTY = "neutral_empty"        # 双方都不完全包围的空区
INTERIOR_STONE_STANDARD = "interior_stone_standard"
INTERIOR_STONE_EDGE = "interior_stone_edge"
INTERIOR_STONE_CORNER = "interior_stone_corner"
OPPONENT_STONE = "opponent_stone"


def interior_kind(target: Board, row: int, col: int) -> Optional[str]:
    """判断 ``(row, col)`` 是否位于本方团块内部，并返回位置类别。

    规则（用户定义）：**该点相邻的全部（在盘内的）棋子都是同色**，即为内部。
    棋盘外面不存在，所以角上只看 2 个邻点、边上只看 3 个邻点、其余看 4 个邻点：
    只要这些邻点全都是本方棋子，就算内部。

    这类点往往是曾经发生提子的地方 —— 本方团块内部的这个子，历史上很可能是对方先
    下在这里、被提掉、再被本方反提，所以"对方曾在此处落子"是合理历史，
    该点处对方的固定代价应当更低。

    返回 ``None``（不是内部）、``"corner"``（角，2 邻点）、``"edge"``（边，3 邻点）
    或 ``"standard"``（常规，4 邻点）。
    """
    colour = target.get(row, col)
    if colour is None:
        return None
    neighbours = 0
    for (nr, nc) in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if not target.in_bounds(nr, nc):
            continue
        if target.get(nr, nc) != colour:
            return None
        neighbours += 1
    if neighbours == 4:
        return "standard"
    if neighbours == 3:
        return "edge"
    if neighbours == 2:
        return "corner"
    return None  # 1x1 之类退化棋盘，不视为内部


@dataclass
class TerminalLossTable:
    """黑白各一张固定点位代价表，附带用于解释罚分的空区信息。"""

    size: int
    black_map: list[list[float]]      # black_map[row][col]：黑方在该点落子的终盘代价
    white_map: list[list[float]]
    region_id: list[list[int]]        # 每个空点所属空区编号；棋子点为 -1
    regions: list[dict] = field(default_factory=list)
    config: Optional[TerminalLossConfig] = None

    def map_for(self, colour: str) -> list[list[float]]:
        if colour == BLACK:
            return self.black_map
        if colour == WHITE:
            return self.white_map
        raise ValueError(f"非法颜色: {colour!r}")

    def loss(self, colour: str, row: int, col: int) -> float:
        return self.map_for(colour)[row][col]

    def flat(self, colour: str) -> list[float]:
        return [v for row in self.map_for(colour) for v in row]

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "point_order": "row-major, row 0 = top, col 0 = left",
            "black_map": self.black_map,
            "white_map": self.white_map,
            "region_id": self.region_id,
            "regions": self.regions,
            "config": vars(self.config) if self.config else None,
        }


def opponent_territory_penalty(area: int, cfg: TerminalLossConfig) -> float:
    """对方完全包围的空区里落子的罚分，随空区面积增长。

    默认用**指数**形式::

        base * exp(k * area) - offset

    当前配置为 ``base = 1.0``、``k = 0.12``、``offset = 1.0``，即
    ``exp(0.12 * area) - 1``：面积 0 时为 0，随面积指数增长，**不设截断上限**
    （由 ``max_penalty`` 控制，设为 0 或负数即表示不截断）。
    结果下限为 0（不允许负代价）。也支持 ``mode = "linear"`` 作对照。
    """
    if cfg.mode == "linear":
        value = cfg.base_temp_penalty + cfg.area_coefficient * area - cfg.offset
    else:
        value = cfg.base_temp_penalty * math.exp(cfg.area_coefficient * area) - cfg.offset
    value = max(0.0, value)
    if cfg.max_penalty is not None and cfg.max_penalty > 0:
        value = min(value, cfg.max_penalty)
    return value


def build_terminal_loss(target: Board,
                        config: Optional[TerminalLossConfig] = None) -> TerminalLossTable:
    """由目标棋盘生成黑白固定点位代价表。"""
    cfg = config or TerminalLossConfig()
    size = target.size
    black_map = [[0.0] * size for _ in range(size)]
    white_map = [[0.0] * size for _ in range(size)]
    region_id = [[-1] * size for _ in range(size)]

    regions = target.empty_regions()
    region_records: list[dict] = []
    for idx, region in enumerate(regions):
        points = region["points"]
        area = region["area"]
        black_border = region["black_border"]
        white_border = region["white_border"]
        # 只接触单一颜色才算"被该方完全包围"
        enclosed_by_black = black_border and not white_border
        enclosed_by_white = white_border and not black_border

        # 黑方在该空区落子：被白方（对方）包围 -> 按面积曲线取罚分；
        # 被黑方自己包围、或中立空区 -> 一律取很大的有限罚分
        black_penalty = (opponent_territory_penalty(area, cfg)
                         if enclosed_by_white else cfg.closed_empty_penalty)
        # 白方在该空区落子：对称处理
        white_penalty = (opponent_territory_penalty(area, cfg)
                         if enclosed_by_black else cfg.closed_empty_penalty)
        for (row, col) in points:
            region_id[row][col] = idx
            black_map[row][col] = black_penalty
            white_map[row][col] = white_penalty
        region_records.append({
            "id": idx,
            "area": area,
            "black_border": black_border,
            "white_border": white_border,
            "enclosed_by_black": enclosed_by_black,
            "enclosed_by_white": enclosed_by_white,
            "black_penalty": black_penalty,
            "white_penalty": white_penalty,
            "points": [list(p) for p in sorted(points)],
        })

    for row in range(size):
        for col in range(size):
            colour = target.get(row, col)
            if colour is None:
                continue
            # 该点处"对方"的代价：位于本方团块内部时取更低的对应档位
            kind = interior_kind(target, row, col)
            if kind == "standard":
                opponent_penalty = cfg.interior_stone_penalty_standard
            elif kind == "edge":
                opponent_penalty = cfg.interior_stone_penalty_edge
            elif kind == "corner":
                opponent_penalty = cfg.interior_stone_penalty_corner
            else:
                opponent_penalty = cfg.opponent_stone_penalty
            if colour == BLACK:
                # 黑方在终盘已是黑子的点落子：同色，代价 0；白方在该点落子：异色点
                black_map[row][col] = 0.0
                white_map[row][col] = opponent_penalty
            else:
                white_map[row][col] = 0.0
                black_map[row][col] = opponent_penalty

    return TerminalLossTable(size=size, black_map=black_map, white_map=white_map,
                             region_id=region_id, regions=region_records, config=cfg)


def classify(table: TerminalLossTable, target: Board, colour: str, row: int, col: int) -> str:
    """返回该点对应该颜色的分类标签，便于解释罚分来源。"""
    colour_at_target = target.get(row, col)
    if colour_at_target == colour:
        return SAME_COLOUR
    if colour_at_target is not None:
        kind = interior_kind(target, row, col)
        if kind == "standard":
            return INTERIOR_STONE_STANDARD
        if kind == "edge":
            return INTERIOR_STONE_EDGE
        if kind == "corner":
            return INTERIOR_STONE_CORNER
        return OPPONENT_STONE
    rid = table.region_id[row][col]
    if rid < 0:
        return CLOSED_EMPTY
    region = table.regions[rid]
    enclosed_by_self = (region["enclosed_by_black"] if colour == BLACK
                        else region["enclosed_by_white"])
    enclosed_by_opponent = (region["enclosed_by_white"] if colour == BLACK
                            else region["enclosed_by_black"])
    if enclosed_by_opponent:
        return OPPONENT_TERRITORY
    if enclosed_by_self:
        return CLOSED_EMPTY
    return NEUTRAL_EMPTY
