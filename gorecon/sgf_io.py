"""SGF 输入输出：读取局面/棋谱，导出局面与候选棋谱。

关键约束（项目说明第 3 节）：

* 首版只接受"恰好一局、一个摆子节点"的局面 SGF；遇到 B/W 着手或子树时明确报错，
  防止把答案棋谱误当输入。
* 校验坐标、棋盘大小、黑白重叠与局面合法性，不依赖解析库悄悄修改目标盘面。
* 只有明确出现 ``PL`` 时才把轮次当作已知信息；缺失视为未知。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sgfmill.sgf as sgf

from .board import BLACK, WHITE, Board, IllegalMove, point_to_sgf, sgf_to_point

DEFAULT_KOMI = 7.5
DEFAULT_RULES = "chinese"

COMMENT_PREFIX = "gorecon"


class SgfError(ValueError):
    """SGF 内容不符合本项目输入/输出约定。"""


@dataclass
class Position:
    """一个只含棋子分布的局面。"""

    board: Board
    size: int = 9
    komi: float = DEFAULT_KOMI
    rules: str = DEFAULT_RULES
    player_to_move: Optional[str] = None  # None 表示未知
    known_moves: Optional[int] = None
    source: Optional[str] = None
    raw_properties: dict = field(default_factory=dict)

    def stone_counts(self) -> tuple[int, int]:
        return self.board.stone_count(BLACK), self.board.stone_count(WHITE)


@dataclass
class GameRecord:
    """一份完整对局棋谱（用于生成局面、评测与 KataGo 逐手分析）。"""

    moves: list[tuple[str, tuple[int, int]]]
    size: int = 9
    komi: float = DEFAULT_KOMI
    rules: str = DEFAULT_RULES
    setup_black: list[tuple[int, int]] = field(default_factory=list)
    setup_white: list[tuple[int, int]] = field(default_factory=list)
    player_to_move: Optional[str] = None
    source: Optional[str] = None
    root_properties: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.moves)

    def replay(self, upto: Optional[int] = None) -> tuple[Board, list[list[tuple[int, int]]]]:
        """从初始摆子开始重放前 ``upto`` 手（``None`` 表示全部）。

        返回 ``(棋盘, 每手被提点列表)``；遇到非法着手抛 :class:`SgfError`。
        """
        from .board import board_from_points

        n = len(self.moves) if upto is None else max(0, min(upto, len(self.moves)))
        board = board_from_points(self.setup_black, self.setup_white, self.size)
        captured: list[list[tuple[int, int]]] = []
        for i, (colour, (row, col)) in enumerate(self.moves[:n]):
            try:
                board, taken = board.play(row, col, colour)
            except IllegalMove as exc:
                raise SgfError(f"棋谱第 {i + 1} 手 {colour} {point_to_sgf(row, col)} 非法: {exc}") from exc
            captured.append(taken)
        return board, captured

    def position_at(self, move_number: int) -> Board:
        """返回第 ``move_number`` 手**之前**的盘面（1 基；0 为初始盘面）。"""
        board, _ = self.replay(upto=move_number)
        return board

    def position_after(self, move_number: int) -> Board:
        """返回第 ``move_number`` 手**之后**的盘面（1 基）。"""
        return self.position_at(move_number + 1 if move_number >= 0 else 0)


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------

def _safe_get(node: sgf.Tree_node, identifier: str, default=None):
    try:
        value = node.get(identifier)
    except (KeyError, ValueError, TypeError):
        return default
    return default if value is None else value


def _as_text(value):
    """sgfmill 对某些属性返回 bytes（例如着手坐标 ``b'ee'``），统一转成 str。"""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    return value


# --------------------------------------------------------------------------
# sgfmill 坐标适配
#
# 教训：不要把 sgfmill 的 ``Board`` / ``get_setup_stones`` 返回的整数元组直接当作
# "(行, 列)"使用。它的轴向与本项目内部约定（行从上往下、列从左往右）不一致，直接
# 使用会把盘面**转置**。因此这里只把 sgfmill 当作 SGF 语法解析器：摆子与着手都从
# SGF 坐标字符串（``AB``/``AW`` 的原始值、着手坐标）转成内部坐标。
# --------------------------------------------------------------------------

def _split_setup(node: sgf.Tree_node, size: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """返回内部坐标的 ``(黑摆子, 白摆子)``，来源是 ``AB``/``AW`` 的原始 SGF 值。"""
    black: list[tuple[int, int]] = []
    white: list[tuple[int, int]] = []
    for identifier, target in (("AB", black), ("AW", white)):
        if not node.has_property(identifier):
            continue
        for raw in node.get_raw_list(identifier):
            text = _as_text(raw)
            if not text:
                continue  # AB[]/AW[] 为空表示清除该色摆子
            target.append(sgf_to_point(text, size))
    return sorted(black), sorted(white)


def _main_sequence_nodes(root: sgf.Tree_node) -> list[sgf.Tree_node]:
    """沿主变化（每一步都取第一个子节点）收集节点。"""
    nodes = [root]
    node = root
    while True:
        children = list(node)
        if not children:
            break
        node = children[0]
        nodes.append(node)
    return nodes


def _count_branch_nodes(root: sgf.Tree_node) -> int:
    """统计拥有多于一个子节点的节点数量（即变化/子树个数）。"""
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        children = list(node)
        if len(children) > 1:
            count += 1
        stack.extend(children)
    return count


def _validate_position_board(board: Board, source: str) -> None:
    """校验摆子局面本身是否合法（不允许存在无气棋块）。"""
    checked: set[tuple[int, int]] = set()
    for row in range(board.size):
        for col in range(board.size):
            if (row, col) in checked:
                continue
            colour = board.get(row, col)
            if colour is None:
                continue
            stones, liberties = board.group(row, col)
            checked |= stones
            if liberties == 0:
                pts = ", ".join(point_to_sgf(r, c) for r, c in sorted(stones))
                raise SgfError(f"{source}: 存在无气的{colour}棋块（{pts}），局面本身不合法")


def _game_from_bytes(data: bytes, source: Optional[str] = None) -> sgf.Sgf_game:
    try:
        return sgf.Sgf_game.from_bytes(data)
    except Exception as exc:  # sgfmill 抛出的异常类型较多，统一包装
        raise SgfError(f"无法解析 SGF（{source or '内存数据'}）: {exc}") from exc


def read_sgf_file(path: str | Path) -> sgf.Sgf_game:
    p = Path(path)
    if not p.is_file():
        raise SgfError(f"SGF 文件不存在: {p}")
    return _game_from_bytes(p.read_bytes(), source=str(p))


def load_game_record(path: str | Path) -> GameRecord:
    """读取完整棋谱（含 B/W 着手），用于数据制作与逐手分析。"""
    game = read_sgf_file(path)
    size = game.get_size()
    root = game.get_root()
    try:
        komi = float(game.get_komi())
    except (KeyError, ValueError):
        komi = DEFAULT_KOMI
    rules = str(_safe_get(root, "RU", DEFAULT_RULES) or DEFAULT_RULES).lower()

    setup_black, setup_white = _split_setup(root, size)
    moves: list[tuple[str, tuple[int, int]]] = []
    for node in game.get_main_sequence()[1:]:
        raw = node.get_raw_move()
        if raw is None:
            continue
        colour, sgf_point = raw
        if colour not in (BLACK, WHITE):
            raise SgfError(f"未知着手颜色: {colour!r}")
        sgf_point = _as_text(sgf_point)
        if sgf_point in (None, "", b""):
            raise SgfError("棋谱中出现了 pass；本项目只搜索落子，暂不支持 pass 作为重建目标")
        moves.append((colour, sgf_to_point(sgf_point, size)))

    pl = _safe_get(root, "PL")
    pl = pl if pl in (BLACK, WHITE) else None
    return GameRecord(
        moves=moves,
        size=size,
        komi=komi,
        rules=rules,
        setup_black=setup_black,
        setup_white=setup_white,
        player_to_move=pl,
        source=str(path),
        root_properties={
            "PB": _safe_get(root, "PB"),
            "PW": _safe_get(root, "PW"),
            "DT": _safe_get(root, "DT"),
            "RU": _safe_get(root, "RU"),
        },
    )


def load_position(path: str | Path, *, expected_size: Optional[int] = 9,
                  expected_komi: Optional[float] = None,
                  expected_rules: Optional[str] = None) -> Position:
    """读取单节点局面 SGF，并做严格校验。

    * 必须恰好一个节点、无子树；
    * 不得包含 B/W 着手（否则说明这是棋谱而不是局面）；
    * 校验尺寸、黑白重叠、坐标合法性；
    * 与配置的规则/贴目冲突时报错，不静默采用不一致的规则。
    """
    game = read_sgf_file(path)
    size = game.get_size()
    root = game.get_root()

    if _count_branch_nodes(root) > 0:
        raise SgfError(f"{path}: 检测到子树/多分支，本项目只接受恰好一局的单节点局面")
    nodes = _main_sequence_nodes(root)
    if len(nodes) != 1:
        raise SgfError(f"{path}: 检测到 {len(nodes)} 个节点，只接受恰好一个摆子节点；"
                       f"若这是完整棋谱，请先用 from_manual_to_situation 转换为局面")
    if root.has_property("B") or root.has_property("W"):
        raise SgfError(f"{path}: 含 B/W 着手属性，这是棋谱而不是局面；请先转换为局面再作为输入")
    if not root.has_property("AB") and not root.has_property("AW"):
        raise SgfError(f"{path}: 局面为空（无 AB/AW），请检查输入")

    if expected_size is not None and size != expected_size:
        raise SgfError(f"{path}: 棋盘尺寸为 {size}，与配置的 {expected_size} 不一致")

    black, white = _split_setup(root, size)
    board = Board(size)
    occupied: dict[tuple[int, int], str] = {}
    for colour, points in ((BLACK, black), (WHITE, white)):
        for (row, col) in points:
            if (row, col) in occupied:
                raise SgfError(f"{path}: 黑白棋子重叠于点 {point_to_sgf(row, col)}")
            occupied[(row, col)] = colour
            board.set_stone(row, col, colour)

    _validate_position_board(board, str(path))

    try:
        komi = float(game.get_komi())
    except (KeyError, ValueError):
        komi = DEFAULT_KOMI
    rules = _safe_get(root, "RU")
    if expected_komi is not None and abs(komi - expected_komi) > 1e-9:
        raise SgfError(f"{path}: 贴目为 {komi}，与配置的 {expected_komi} 不一致；请显式对齐后再运行")
    if expected_rules is not None and rules is not None:
        if str(rules).strip().lower() != str(expected_rules).strip().lower():
            raise SgfError(f"{path}: 规则为 {rules!r}，与配置的 {expected_rules!r} 不一致")

    pl = _safe_get(root, "PL")
    pl = pl if pl in (BLACK, WHITE) else None

    return Position(
        board=board,
        size=size,
        komi=komi,
        rules=str(rules).strip().lower() if rules else DEFAULT_RULES,
        player_to_move=pl,
        source=str(path),
        raw_properties={"AB": len(black), "AW": len(white)},
    )


def position_from_record(record: GameRecord, move_number: Optional[int] = None) -> Position:
    """把棋谱（或其前 ``move_number`` 手后的盘面）转成只含终盘的 :class:`Position`。"""
    n = len(record.moves) if move_number is None else move_number
    board, _ = record.replay(upto=n)
    pl = None
    if n < len(record.moves):
        pl = record.moves[n][0]
    elif n > 0:
        pl = WHITE if record.moves[-1][0] == BLACK else BLACK
    return Position(
        board=board,
        size=record.size,
        komi=record.komi,
        rules=record.rules,
        player_to_move=pl,
        known_moves=n,
        source=record.source,
    )


# --------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------

def position_to_sgf_text(position: Position, *, comment: Optional[str] = None) -> str:
    """把局面导出为标准化单节点摆子 SGF（FF[4]、UTF-8、AB/AW）。"""
    size = position.size
    black = [point_to_sgf(r, c) for c in range(size) for r in range(size)
             if position.board.get(r, c) == BLACK]
    white = [point_to_sgf(r, c) for c in range(size) for r in range(size)
             if position.board.get(r, c) == WHITE]

    parts = [
        "(;FF[4]GM[1]CA[UTF-8]",
        f"SZ[{size}]",
        f"RU[{position.rules.upper() if position.rules else 'Chinese'}]",
        f"KM[{_fmt_komi(position.komi)}]",
    ]
    if position.player_to_move:
        parts.append(f"PL[{position.player_to_move.upper()}]")
    if comment:
        parts.append(f"C[{_escape(comment)}]")
    if black:
        parts.append("AB" + "".join(f"[{p}]" for p in black))
    if white:
        parts.append("AW" + "".join(f"[{p}]" for p in white))
    parts.append(")\n")
    return "".join(parts)


def write_position(path: str | Path, position: Position, *, comment: Optional[str] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(position_to_sgf_text(position, comment=comment), encoding="utf-8")
    return p


def moves_to_sgf_text(moves: list[tuple[str, tuple[int, int]]], *, size: int = 9,
                      komi: float = DEFAULT_KOMI, rules: str = DEFAULT_RULES,
                      comment: Optional[str] = None) -> str:
    """把落子序列导出为候选棋谱 SGF。"""
    parts = [
        "(;FF[4]GM[1]CA[UTF-8]",
        f"SZ[{size}]",
        f"RU[{rules.upper() if rules else 'Chinese'}]",
        f"KM[{_fmt_komi(komi)}]",
    ]
    if comment:
        parts.append(f"C[{_escape(comment)}]")
    for colour, (row, col) in moves:
        parts.append(f";{colour.upper()}[{point_to_sgf(row, col)}]")
    parts.append(")\n")
    return "".join(parts)


def write_moves(path: str | Path, moves: list[tuple[str, tuple[int, int]]], **kwargs) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(moves_to_sgf_text(moves, **kwargs), encoding="utf-8")
    return p


def _fmt_komi(komi: float) -> str:
    return f"{komi:g}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("]", "\\]")
