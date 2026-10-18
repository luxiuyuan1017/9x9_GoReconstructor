"""9x9 围棋棋盘：合法性、提子、空区连通与坐标适配。

内部坐标统一约定（与项目说明第 9 节一致）：

* ``(row, col)``，``row`` 从上往下 0..size-1，``col`` 从左往右 0..size-1。
* 第 0 行第 0 列位于左上角。

本模块只负责规则执行，不包含任何 loss 计算。
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Optional

BLACK = "b"
WHITE = "w"
EMPTY = None

COLOURS = (BLACK, WHITE)

GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"  # GTP 列名跳过 I
SGF_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------
# 坐标适配：外部表示 <-> 内部 (row, col)
# --------------------------------------------------------------------------

def sgf_to_point(s: str, size: int = 9) -> tuple[int, int]:
    """SGF 坐标 ``"aa"`` -> ``(0, 0)``（左上）。"""
    if not isinstance(s, str) or len(s) != 2:
        raise ValueError(f"非法 SGF 坐标: {s!r}")
    try:
        col = SGF_ALPHABET.index(s[0])
        row = SGF_ALPHABET.index(s[1])
    except ValueError as exc:  # pragma: no cover - 防御性
        raise ValueError(f"非法 SGF 坐标: {s!r}") from exc
    if not (0 <= col < size and 0 <= row < size):
        raise ValueError(f"SGF 坐标 {s!r} 超出 {size}x{size} 棋盘")
    return row, col


def point_to_sgf(row: int, col: int) -> str:
    """``(0, 0)`` -> ``"aa"``。"""
    return SGF_ALPHABET[col] + SGF_ALPHABET[row]


def gtp_to_point(s: str, size: int = 9) -> tuple[int, int]:
    """GTP 坐标 ``"A9"`` (左上) -> ``(0, 0)``。"""
    s = s.strip().upper()
    if len(s) < 2 or s[0] not in GTP_COLUMNS:
        raise ValueError(f"非法 GTP 坐标: {s!r}")
    col = GTP_COLUMNS.index(s[0])
    row = size - int(s[1:])
    if not (0 <= col < size and 0 <= row < size):
        raise ValueError(f"GTP 坐标 {s!r} 超出 {size}x{size} 棋盘")
    return row, col


def point_to_gtp(row: int, col: int, size: int = 9) -> str:
    """``(0, 0)`` -> ``"A9"``。"""
    return f"{GTP_COLUMNS[col]}{size - row}"


def policy_index_to_point(index: int, size: int = 9) -> Optional[tuple[int, int]]:
    """KataGo policy 向量下标 -> 内部坐标；最后一项 (size*size) 是 pass，返回 ``None``。"""
    if index == size * size:
        return None
    if not 0 <= index < size * size:
        raise ValueError(f"policy 下标越界: {index}")
    # policy 盘面部分按从左上到右下的行优先顺序排列
    return divmod(index, size)


def point_to_policy_index(row: int, col: int, size: int = 9) -> int:
    return row * size + col


# --------------------------------------------------------------------------
# 棋盘
# --------------------------------------------------------------------------

class IllegalMove(ValueError):
    """非法着手（占用、自杀、打劫等）。"""


class Board:
    """不可变式使用的围棋棋盘（``play`` 返回新对象，但内部就地进行提子）。

    使用扁平 ``bytearray`` 存储，取值 ``"b"`` / ``"w"`` / 0。
    """

    __slots__ = ("size", "_cells", "_hash")

    def __init__(self, size: int = 9) -> None:
        if size < 2:
            raise ValueError("棋盘尺寸过小")
        self.size = size
        self._cells = bytearray(size * size)
        self._hash: Optional[bytes] = None

    # -- 基本访问 ---------------------------------------------------------

    @property
    def num_points(self) -> int:
        return self.size * self.size

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def _index(self, row: int, col: int) -> int:
        if not self.in_bounds(row, col):
            raise IndexError(f"坐标越界: ({row}, {col})")
        return row * self.size + col

    def get(self, row: int, col: int) -> Optional[str]:
        """返回 ``"b"`` / ``"w"`` / ``None``。"""
        v = self._cells[self._index(row, col)]
        return None if v == 0 else chr(v)

    def is_empty(self, row: int, col: int) -> bool:
        return self._cells[self._index(row, col)] == 0

    def set_stone(self, row: int, col: int, colour: Optional[str]) -> None:
        """直接摆放/清除棋子，不做提子。"""
        if colour not in (None, BLACK, WHITE):
            raise ValueError(f"非法颜色: {colour!r}")
        self._cells[self._index(row, col)] = 0 if colour is None else ord(colour)
        self._hash = None

    def occupied_points(self) -> list[tuple[Optional[str], tuple[int, int]]]:
        """按 sgfmill 风格返回 ``[(colour, (row, col)), ...]``，顺序为行优先。"""
        out = []
        for row in range(self.size):
            for col in range(self.size):
                c = self.get(row, col)
                if c is not None:
                    out.append((c, (row, col)))
        return out

    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.size = self.size
        other._cells = bytearray(self._cells)
        other._hash = self._hash
        return other

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and other.size == self.size and other._cells == self._cells

    def position_hash(self) -> bytes:
        """盘面指纹（只与棋子分布有关，与历史无关，用于 superko 判重）。"""
        if self._hash is None:
            self._hash = bytes(self._cells)
        return self._hash

    # -- 棋块与气 ---------------------------------------------------------

    def _neighbours(self, row: int, col: int) -> Iterator[tuple[int, int]]:
        size = self.size
        if row > 0:
            yield row - 1, col
        if row < size - 1:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col < size - 1:
            yield row, col + 1

    def group(self, row: int, col: int) -> tuple[set[tuple[int, int]], int]:
        """返回该点所属棋块的所有点和它的气数；空点返回 ``(set(), 0)``。"""
        colour = self.get(row, col)
        if colour is None:
            return set(), 0
        stones: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(row, col)]
        stones.add((row, col))
        while stack:
            r, c = stack.pop()
            for nr, nc in self._neighbours(r, c):
                v = self.get(nr, nc)
                if v is None:
                    liberties.add((nr, nc))
                elif v == colour and (nr, nc) not in stones:
                    stones.add((nr, nc))
                    stack.append((nr, nc))
        return stones, len(liberties)

    # -- 落子 -------------------------------------------------------------

    def place_stone(self, row: int, col: int, colour: str) -> list[tuple[int, int]]:
        """在当前棋盘上直接落子并提子（**破坏性**操作），返回被提点列表。

        只做"落子 + 提掉无气的对方棋块"，不判断自杀；自杀判断由 :meth:`play` /
        :meth:`is_legal` 在这之后用 :meth:`group` 检查。

        注意：被提的棋块是**紧邻落子点的对方棋块中已经没有气的那些**。对方棋块的
        最后一气不一定与落子点相邻（例如黑在 (0,2) 提掉白 (0,0)，其气是 (0,1)(1,0)），
        因此必须先在落子点上完成落子，再检查所有相邻的对方棋块。
        """
        if colour not in COLOURS:
            raise ValueError(f"非法颜色: {colour!r}")
        self._cells[self._index(row, col)] = ord(colour)
        self._hash = None
        opponent = WHITE if colour == BLACK else BLACK
        removed: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for nr, nc in self._neighbours(row, col):
            if (nr, nc) in seen or self.get(nr, nc) != opponent:
                continue
            stones, liberties = self.group(nr, nc)
            seen |= stones
            if liberties == 0:
                for sr, sc in stones:
                    self._cells[self._index(sr, sc)] = 0
                removed.extend(sorted(stones))
        if removed:
            self._hash = None
        return removed

    def _position_after(self, row: int, col: int, colour: str) -> tuple["Board", list[tuple[int, int]], int]:
        """落子后的候选棋盘、被提点、落子棋块的气数（不做自杀判断）。"""
        trial = self.copy()
        captured = trial.place_stone(row, col, colour)
        _, liberties = trial.group(row, col)
        return trial, captured, liberties

    def is_legal(self, row: int, col: int, colour: str) -> bool:
        """当前盘面上该颜色能否在此落子（不含打劫/superko 判断）。"""
        if colour not in COLOURS:
            raise ValueError(f"非法颜色: {colour!r}")
        if not self.in_bounds(row, col):
            return False
        if not self.is_empty(row, col):
            return False
        _trial, _captured, liberties = self._position_after(row, col, colour)
        return liberties > 0  # 无气且无提子则视为自杀

    def play(self, row: int, col: int, colour: str) -> tuple["Board", list[tuple[int, int]]]:
        """落子并提子，返回新的棋盘与被提点列表；非法着手抛 :class:`IllegalMove`。

        打劫（superko）不在此处判断，由搜索节点用 :meth:`position_hash` 维护。
        """
        if colour not in COLOURS:
            raise ValueError(f"非法颜色: {colour!r}")
        if not self.in_bounds(row, col):
            raise IllegalMove(f"坐标越界: ({row}, {col})")
        if not self.is_empty(row, col):
            raise IllegalMove(f"({row}, {col}) 已有棋子")
        new, captured, liberties = self._position_after(row, col, colour)
        if liberties == 0:
            raise IllegalMove(f"({row}, {col}) 为自杀手")
        return new, captured

    def legal_points(self, colour: str) -> list[tuple[int, int]]:
        """当前所有合法交叉点（不含 pass），行优先。"""
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.is_legal(row, col, colour)
        ]

    # -- 空区连通 ---------------------------------------------------------

    def empty_regions(self) -> list[dict]:
        """按上下左右连通划分完整空区。

        返回 ``[{"points": [...], "area": int, "black_border": bool, "white_border": bool}, ...]``，
        其中 ``*_border`` 表示该空区是否与对应颜色的棋子相邻。棋盘边缘只充当边界，
        不计入任何颜色。
        """
        visited = [[False] * self.size for _ in range(self.size)]
        regions = []
        for row in range(self.size):
            for col in range(self.size):
                if visited[row][col] or not self.is_empty(row, col):
                    continue
                points: list[tuple[int, int]] = []
                black_border = white_border = False
                queue = deque([(row, col)])
                visited[row][col] = True
                while queue:
                    r, c = queue.popleft()
                    points.append((r, c))
                    for nr, nc in self._neighbours(r, c):
                        v = self.get(nr, nc)
                        if v is None:
                            if not visited[nr][nc]:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                        elif v == BLACK:
                            black_border = True
                        else:
                            white_border = True
                regions.append(
                    {
                        "points": points,
                        "area": len(points),
                        "black_border": black_border,
                        "white_border": white_border,
                    }
                )
        return regions

    # -- 统计 -------------------------------------------------------------

    def stone_count(self, colour: str) -> int:
        target = ord(colour)
        return sum(1 for v in self._cells if v == target)

    def diff_points(self, other: "Board") -> list[tuple[int, int, Optional[str], Optional[str]]]:
        """返回与本盘面不同的点 ``(row, col, self_colour, other_colour)``。"""
        if other.size != self.size:
            raise ValueError("棋盘尺寸不一致")
        out = []
        for row in range(self.size):
            for col in range(self.size):
                a, b = self.get(row, col), other.get(row, col)
                if a != b:
                    out.append((row, col, a, b))
        return out

    def same_position(self, other: "Board") -> bool:
        return self.size == other.size and self._cells == other._cells

    # -- 展示 -------------------------------------------------------------

    def ascii(self) -> str:
        lines = []
        for row in range(self.size):
            cells = []
            for col in range(self.size):
                c = self.get(row, col)
                cells.append("." if c is None else ("X" if c == BLACK else "O"))
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Board {self.size}x{self.size} b={self.stone_count(BLACK)} w={self.stone_count(WHITE)}>"


def board_from_points(black: Iterable[tuple[int, int]], white: Iterable[tuple[int, int]], size: int = 9) -> Board:
    """由黑白点位构造棋盘，重叠点会报错（不做静默覆盖）。"""
    board = Board(size)
    seen: dict[tuple[int, int], str] = {}
    for colour, points in ((BLACK, black), (WHITE, white)):
        for (row, col) in points:
            if not board.in_bounds(row, col):
                raise ValueError(f"点位越界: ({row}, {col})")
            if (row, col) in seen:
                raise ValueError(f"黑白棋子重叠于点 ({row}, {col})")
            seen[(row, col)] = colour
            board.set_stone(row, col, colour)
    return board
