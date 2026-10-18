"""主程序的按层启发式搜索。

严格按项目说明第 6、7 节，以及 main/main.py 的补充要求：

* 累计代价：``C(H ⊕ (c,x)) = C(H) + alpha * (-log p_K(x|H,c)) + lambda * L_终盘^c(x)``
  （``alpha`` 是 KataGo 项权重，``lambda`` 是终盘项权重；都取 1 即说明里的原始定义）；
* 每层把**所有**父候选展开出的合法子节点合并成**一个池**，按统一标准保留 n 条
  （n 是全局每层上限，不是每个父节点各留 n 条）；
* main.py 指定的排序标准是**每步平均 loss**，即 ``总代价 / 已下手数``
  （这样不会因为路径短就占便宜）；
* ``progress_weight`` 额外把"与目标终盘不一致的点数"折算进排序键，用来把搜索引向终盘；
  它**不计入**报告出来的 loss 数值，只影响谁被留下；
* 某条路径超过 ``max_moves`` 手即杀死该路径；
* 盘面与目标终盘逐点一致即成功，成功候选另存、不再扩展；
* 同一盘面可由不同历史达到，因此候选按完整历史区分，不按盘面哈希合并。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .board import BLACK, WHITE, Board, IllegalMove
from .katago_loss import KataGoEngine, KataGoError
from .terminal_loss import TerminalLossTable

#: 目标终盘还缺某色棋子时认为"该色还要落子"；仅用于排序启发，不参与 loss
_MISSING_WEIGHT = 1.0


@dataclass
class PathCandidate:
    """一条候选棋谱：棋盘 + 完整历史 + 累计代价。

    单步代价按项目说明第 6 节，其中 KataGo 项带独立权重 ``alpha``：

        C(H⊕(c,x)) = C(H) + alpha * (-log p_K(x|H,c)) + lambda * L_终盘^c(x)

    ``alpha = 1``、``lambda = 1`` 时就是说明里的原始定义。
    """

    board: Board
    moves: tuple[tuple[str, tuple[int, int]], ...]
    colour_to_move: str
    cumulative_katago: float = 0.0        # 已乘 alpha
    cumulative_terminal: float = 0.0      # 已乘 lambda
    matched_points: int = 0
    history_hashes: frozenset[bytes] = frozenset()
    step_katago: tuple[float, ...] = ()   # 每手的 KataGo 项（未乘系数）
    step_terminal: tuple[float, ...] = ()  # 每手的终盘项（未乘系数）

    @property
    def total_cost(self) -> float:
        return self.cumulative_katago + self.cumulative_terminal

    @property
    def steps(self) -> int:
        return len(self.moves)

    @property
    def average_loss(self) -> float:
        """每步平均 loss。main.py 的保留标准就是它。"""
        return self.total_cost / self.steps if self.steps else 0.0

    @property
    def raw_katago_sum(self) -> float:
        """未乘 alpha 的 KataGo 项之和，便于看清楚加权前后的差别。"""
        return sum(self.step_katago)

    @property
    def raw_terminal_sum(self) -> float:
        return sum(self.step_terminal)

    def rank_key(self, target_stones: int, progress_weight: float) -> float:
        """排序键：每步平均 loss（+ 可选的"还差多少目标棋子"引导项）。"""
        if progress_weight <= 0.0 or self.steps == 0:
            return self.average_loss
        missing = max(0, target_stones - self.matched_points)
        return self.average_loss + progress_weight * (missing / self.steps)

    def key(self) -> tuple:
        return (self.board.position_hash(), self.moves)


@dataclass
class SearchConfig:
    beam_width: int = 50
    top_k: int = 5
    max_moves: int = 100
    time_budget_s: float = 3600.0
    engine_query_budget: int = 20000
    progress_weight: float = 1.0
    #: KataGo 项的权重 alpha：alpha * (-log p) 计入累计代价。1.0 即说明里的原始定义。
    alpha_katago: float = 1.0
    #: 终盘项的权重 lambda：lambda * L_终盘 计入累计代价。
    lambda_terminal: float = 1.0


@dataclass
class SearchOutcome:
    """搜索结果：是否成功、成功路径、日志与统计。"""

    found: bool
    paths: list[PathCandidate] = field(default_factory=list)
    stop_reason: str = ""
    layers: int = 0
    expanded: int = 0
    engine_queries: int = 0
    killed_over_max_moves: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[PathCandidate]:
        return self.paths[0] if self.paths else None


def matched_point_count(board: Board, target: Board) -> int:
    """已摆对的棋子数：黑白分别统计"该色棋子出现在目标终盘同色位置上"的个数。

    只数棋子，不数空点 —— 空点多并不代表接近终盘，用棋子数才能反映进度。
    """
    if board.size != target.size:
        raise ValueError("棋盘尺寸不一致")
    count = 0
    for row in range(board.size):
        for col in range(board.size):
            colour = board.get(row, col)
            if colour is not None and colour == target.get(row, col):
                count += 1
    return count


def matches_target(board: Board, target: Board) -> bool:
    return board.same_position(target)


def heuristic_search(position,
                     terminal: TerminalLossTable,
                     engine: Optional[KataGoEngine],
                     config: SearchConfig,
                     *,
                     target: Optional[Board] = None) -> SearchOutcome:
    """执行按层启发式搜索。

    ``position`` 提供棋盘尺寸、规则与贴目；``target`` 为目标终盘（默认取 ``position.board``）。
    ``engine`` 为 ``None`` 时不查询 KataGo（只用于逻辑测试，代价里没有 KataGo 项）。
    """
    target_board = target if target is not None else position.board
    size = position.size
    rules = position.rules or "chinese"
    komi = position.komi
    target_stones = target_board.stone_count(BLACK) + target_board.stone_count(WHITE)

    log_lines: list[str] = []

    def emit(message: str) -> None:
        log_lines.append(message)
        print(message, flush=True)

    deadline = time.time() + config.time_budget_s if config.time_budget_s > 0 else float("inf")
    engine_queries_before = engine.engine_calls if engine is not None else 0

    root_board = Board(size)
    root = PathCandidate(
        board=root_board,
        moves=(),
        colour_to_move=BLACK,
        matched_points=matched_point_count(root_board, target_board),
        history_hashes=frozenset({root_board.position_hash()}),
    )

    if matches_target(root_board, target_board):
        emit("空盘即等于目标终盘，无需搜索。")
        return SearchOutcome(found=True, paths=[root], stop_reason="空盘即匹配",
                             log=log_lines)

    active: list[PathCandidate] = [root]
    finished: list[PathCandidate] = []
    layers = expanded = killed = 0
    stop_reason = ""

    while active:
        if layers >= config.max_moves:
            stop_reason = f"达到 max_moves={config.max_moves}"
            break
        if time.time() > deadline:
            stop_reason = f"达到时间预算 {config.time_budget_s:.0f}s"
            break
        if engine is not None and engine.engine_calls - engine_queries_before >= config.engine_query_budget:
            stop_reason = f"达到引擎查询预算 {config.engine_query_budget}"
            break

        # --- 每个父候选取一次动态 loss（按完整历史去重，不按盘面合并） ---
        parent_loss: dict[tuple, dict[tuple[int, int], float]] = {}
        if engine is not None:
            for node in active:
                key = node.key()
                if key in parent_loss:
                    continue
                try:
                    result = engine.query_moves(node.moves, rules=rules, komi=komi, size=size)
                except KataGoError as exc:
                    stop_reason = f"KataGo 查询失败: {exc}"
                    emit(stop_reason)
                    active = []
                    break
                floor = engine.loss_config.prob_floor
                parent_loss[key] = {
                    point: -math.log(max(result.policy[point[0] * size + point[1]], floor))
                    for point in result.legal_moves
                }
            if not active:
                break

        # --- 展开所有合法点 ---
        children: list[PathCandidate] = []
        for node in active:
            colour = node.colour_to_move
            table = terminal.map_for(colour)
            losses = parent_loss.get(node.key()) if engine is not None else None
            for (row, col) in node.board.legal_points(colour):
                kg = 0.0 if losses is None else losses.get((row, col), float("inf"))
                try:
                    child_board, _captured = node.board.play(row, col, colour)
                except IllegalMove:
                    continue
                board_hash = child_board.position_hash()
                if board_hash in node.history_hashes:
                    continue  # 位置超级劫：整条历史不重复盘面
                child_terminal_raw = table[row][col]
                child = PathCandidate(
                    board=child_board,
                    moves=node.moves + ((colour, (row, col)),),
                    colour_to_move=WHITE if colour == BLACK else BLACK,
                    cumulative_katago=node.cumulative_katago + config.alpha_katago * kg,
                    cumulative_terminal=(node.cumulative_terminal
                                         + config.lambda_terminal * child_terminal_raw),
                    matched_points=matched_point_count(child_board, target_board),
                    history_hashes=node.history_hashes | {board_hash},
                    step_katago=node.step_katago + (kg,),
                    step_terminal=node.step_terminal + (child_terminal_raw,),
                )
                expanded += 1
                if child.steps > config.max_moves:
                    killed += 1
                    continue
                if matches_target(child_board, target_board):
                    finished.append(child)
                    continue
                children.append(child)

        layers += 1
        # --- 全局统一保留 n 条：main.py 的标准是"每步平均 loss 最小" ---
        children.sort(key=lambda n: (n.rank_key(target_stones, config.progress_weight),
                                     n.total_cost, n.steps))
        active = children[: config.beam_width]
        best_avg = active[0].average_loss if active else float("nan")
        best_match = active[0].matched_points if active else 0
        layer_best_match = max((n.matched_points for n in children), default=0)
        emit(f"第 {layers} 层：展开 {len(children)} 条，保留 {len(active)} 条，"
             f"已命中终盘 {len(finished)} 条，被杀路径 {killed}；"
             f"本层最优 每步平均loss={best_avg:.4f} 已摆对棋子={best_match}/{target_stones}；"
             f"本层最多已摆对棋子={layer_best_match}/{target_stones}")

        if finished:
            stop_reason = f"成功：{len(finished)} 条路径与目标终盘逐点一致"
            break
        if not active:
            stop_reason = "所有路径都已超出 max_moves，候选池耗尽"
            break

    if not stop_reason:
        stop_reason = "活动候选耗尽，预算内未找到与终盘一致的路径"

    finished.sort(key=lambda n: (n.average_loss, n.total_cost))
    return SearchOutcome(
        found=bool(finished),
        paths=finished[: config.top_k],
        stop_reason=stop_reason,
        layers=layers,
        expanded=expanded,
        engine_queries=(engine.engine_calls - engine_queries_before) if engine else 0,
        killed_over_max_moves=killed,
        log=log_lines,
    )


def verify_path(candidate: PathCandidate, target: Board) -> dict:
    """独立重放这条路径，确认最终盘面与目标终盘逐点一致。"""
    board = Board(target.size)
    for i, (colour, (row, col)) in enumerate(candidate.moves):
        try:
            board, _ = board.play(row, col, colour)
        except IllegalMove as exc:  # pragma: no cover - 理论上不会发生
            return {"ok": False, "reason": f"第 {i + 1} 手非法: {exc}"}
    diffs = board.diff_points(target)
    return {
        "ok": not diffs,
        "reason": "逐点一致" if not diffs else f"{len(diffs)} 个点不一致",
        "diff_points": diffs,
        "moves": len(candidate.moves),
    }
