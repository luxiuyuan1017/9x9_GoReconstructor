"""子程序二：动态 KataGo 点位 loss。

采用本地、冻结的 KataGo 的 JSON analysis 接口，使用普通模型的**原始 policy**
（不以胜率、目数、搜索访问次数或一选排名代替落子概率）。

* 一个常驻进程，长驻复用；每条查询传入候选的完整 moves 历史与匹配的规则、贴目；
* 一次局面查询获得全部落点概率，不对每个子落点重新推理；
* ``-log(p)`` 作为动态点位 loss，过小的合法概率用可配置数值下限截断；
* 非法点剔除（policy 中对应项缺失或为 0），pass 项（最后一项）不在 9x9 热图中展开；
* 显式设置 ``ignorePreRootHistory=false``，保留候选历史对预测的影响。

缓存属于可选优化：这里以"模型 + 配置 + 规则 + 贴目 + 轮次 + 完整历史"作为键，
不按盘面合并历史相关评分。
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .board import BLACK, WHITE, Board, point_to_gtp, policy_index_to_point
from .config import EngineConfig, KataGoLossConfig, PROJECT_ROOT
from .sgf_io import Position


class KataGoError(RuntimeError):
    """KataGo 进程启动、通信或返回内容出错。"""


@dataclass
class KataGoResult:
    """一次 KataGo 局面查询的结果。"""

    turn: int
    colour: str                      # 该局面轮到谁落子
    policy: list[float]              # 原始 policy 分布（长度 size*size+1，最后一项为 pass）
    legal_moves: list[tuple[int, int]]
    illegal_moves: list[tuple[int, int]]
    pass_probability: float
    raw: dict = field(default_factory=dict, repr=False)
    elapsed_s: float = 0.0

    def policy_at(self, row: int, col: int, size: int = 9) -> float:
        return self.policy[row * size + col]

    def pointwise_loss(self, size: int, prob_floor: float = 1e-6) -> list[list[float]]:
        """把 policy 转成 9x9 的 ``-log p`` 表；非法点记为 ``math.inf``。"""
        table = [[math.inf] * size for _ in range(size)]
        legal = set(self.legal_moves)
        for (row, col) in legal:
            p = self.policy[row * size + col]
            table[row][col] = -math.log(max(p, prob_floor))
        return table


@dataclass
class KataGoQuery:
    key: str
    moves: list[tuple[str, tuple[int, int]]]
    turn: int
    rules: str
    komi: float
    size: int
    colour: str


class KataGoEngine:
    """常驻 KataGo analysis 进程的客户端。

    用法::

        with KataGoEngine(engine_cfg) as engine:
            result = engine.query_moves(moves, turn=len(moves))
    """

    def __init__(self, config: EngineConfig,
                 loss_config: Optional[KataGoLossConfig] = None,
                 *, verbose: bool = False) -> None:
        self.config = config
        self.loss_config = loss_config or KataGoLossConfig()
        self.verbose = verbose
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_q: "queue.Queue[str]" = queue.Queue()
        self._stderr_lines: list[str] = []
        self._cache: dict[str, KataGoResult] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self.engine_calls = 0
        self.startup_error: Optional[str] = None

    # -- 生命周期 ---------------------------------------------------------

    def __enter__(self) -> "KataGoEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        cfg = self.config
        for label, value in (("katago_path", cfg.katago_path),
                             ("model_path", cfg.model_path),
                             ("analysis_config", cfg.analysis_config)):
            if not Path(value).is_file():
                raise KataGoError(f"{label} 指向的文件不存在: {value}")
        cmd = [cfg.katago_path, "analysis",
               "-model", cfg.model_path,
               "-config", cfg.analysis_config]
        if cfg.overrides:
            overrides = ",".join(f"{k}={_fmt_override(v)}" for k, v in cfg.overrides.items())
            cmd += ["-override-config", overrides]
        # 显式保留候选历史对预测的影响
        analysis_overrides = [f"ignorePreRootHistory={'true' if cfg.ignore_pre_root_history else 'false'}"]
        if cfg.max_visits:
            analysis_overrides.append(f"maxVisits={cfg.max_visits}")
        cmd += ["-override-config", ",".join(analysis_overrides)]

        if self.verbose:
            print("[katago] " + " ".join(cmd))
        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):  # Windows：避免弹出控制台窗口
            creationflags = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            # 以项目根为工作目录：KataGo 的相对数据目录（日志等）都落在本文件夹内
            cwd=str(PROJECT_ROOT),
            creationflags=creationflags,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._wait_ready()

    def _pump_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._stdout_q.put(line)
        self._stdout_q.put("")  # 结束哨兵

    def _pump_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            self._stderr_lines.append(line)
            del self._stderr_lines[:-400]
            if self.verbose:
                print(f"[katago:err] {line}")

    def _wait_ready(self) -> None:
        """等待引擎可用：直到出现 "Analysis Engine starting" 之后的 NN 初始化完成。

        简化策略：先发一条最简查询，等它的回复。首次 OpenCL 自动调优会在这期间完成。
        """
        probe = {
            "id": "startup-probe",
            "moves": [],
            "rules": "chinese",
            "komi": 7.5,
            "boardXSize": 9,
            "boardYSize": 9,
            "analyzeTurns": [0],
            "includePolicy": True,
            "maxVisits": 1,
        }
        self._send(probe)
        deadline = time.time() + self.config.startup_timeout_ms / 1000.0
        try:
            self._await_replies(["startup-probe"], deadline)
        except KataGoError as exc:
            detail = "\n".join(self._stderr_lines[-15:])
            raise KataGoError(f"KataGo 启动失败：{exc}\n--- 引擎日志尾部 ---\n{detail}") from exc

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -- 查询 -------------------------------------------------------------

    def _send(self, payload: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise KataGoError(f"向 KataGo 写入查询失败（进程可能已退出）: {exc}") from exc

    def _await_replies(self, ids: Sequence[str], deadline: float) -> dict[str, dict]:
        wanted = set(ids)
        found: dict[str, dict] = {}
        while wanted:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise KataGoError(f"等待回复超时，未收到 {sorted(wanted)}")
            try:
                line = self._stdout_q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    raise KataGoError(f"KataGo 进程已退出，返回码 {self._proc.returncode}")
                continue
            if line == "":
                raise KataGoError("KataGo 输出流已关闭")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"[katago:skip] {line[:160]}")
                continue
            if msg.get("error"):
                raise KataGoError(f"KataGo 返回错误: {msg['error']}")
            mid = msg.get("id")
            if mid in wanted:
                wanted.discard(mid)
                found[mid] = msg
        return found

    def _next_id(self, prefix: str = "q") -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def query(self, moves: Sequence[tuple[str, tuple[int, int]]], turn: int, *,
              rules: str = "chinese", komi: float = 7.5, size: int = 9,
              colour: Optional[str] = None, use_cache: bool = True) -> KataGoResult:
        """查询"前 ``turn`` 手之后"的局面，返回该局面下轮到落子一方的 policy。"""
        moves = [tuple(m) for m in moves]
        if colour is None:
            colour = BLACK if turn % 2 == 0 else WHITE
        key = self._cache_key(moves, turn, rules, komi, size, colour)
        if use_cache and key in self._cache:
            return self._cache[key]

        payload = {
            "id": self._next_id(),
            "moves": [[c.upper(), point_to_gtp(r, cc, size)] for (c, (r, cc)) in moves[:turn]],
            "rules": rules,
            "komi": komi,
            "boardXSize": size,
            "boardYSize": size,
            "analyzeTurns": [len(moves[:turn])],
            "includePolicy": True,
            "maxVisits": self.config.max_visits,
        }
        with self._lock:
            self._send(payload)
            self.engine_calls += 1
            deadline = time.time() + self.config.query_timeout_ms / 1000.0
            replies = self._await_replies([payload["id"]], deadline)
        msg = replies[payload["id"]]
        result = self._parse(msg, turn=turn, colour=colour, size=size)
        if use_cache:
            self._cache[key] = result
        return result

    def query_moves(self, moves: Sequence[tuple[str, tuple[int, int]]], **kwargs) -> KataGoResult:
        """查询 ``moves`` 落子**之前**的局面（``turn = len(moves)``）。

        注意：每手的 KataGo 分数必须从"落子前"的父历史取得，因此传入父历史即可。
        """
        return self.query(moves, turn=len(moves), **kwargs)

    def query_history(self, moves: Sequence[tuple[str, tuple[int, int]]], turn: int, **kwargs) -> KataGoResult:
        return self.query(moves, turn=turn, **kwargs)

    def _parse(self, msg: dict, *, turn: int, colour: str, size: int) -> KataGoResult:
        policy = msg.get("policy")
        if policy is None:
            raise KataGoError("返回缺少 policy 字段（需确认 includePolicy 已生效）")
        if len(policy) != size * size + 1:
            raise KataGoError(f"policy 长度 {len(policy)} 与棋盘 {size}x{size} 不符")

        legal_moves: list[tuple[int, int]] = []
        # KataGo 对非法点的 policy 项为 0；本地规则仍需独立检查（见 board.Board.is_legal）。
        for idx, p in enumerate(policy[:-1]):
            if p > 0.0:
                point = policy_index_to_point(idx, size)
                if point is not None:
                    legal_moves.append(point)

        legal_set = set(legal_moves)
        illegal = [(r, c) for r in range(size) for c in range(size) if (r, c) not in legal_set]
        return KataGoResult(
            turn=turn,
            colour=colour,
            policy=[float(p) for p in policy],
            legal_moves=sorted(legal_moves),
            illegal_moves=illegal,
            pass_probability=float(policy[-1]),
            raw=msg,
        )

    @staticmethod
    def _cache_key(moves, turn, rules, komi, size, colour) -> str:
        hist = ";".join(f"{c}{point_to_gtp(r, cc, size)}" for (c, (r, cc)) in moves[:turn])
        return f"{size}|{rules}|{komi}|{colour}|{turn}|{hist}"


def _fmt_override(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------
# 面向单点的便捷接口
# --------------------------------------------------------------------------

def compute_katago_loss(engine: KataGoEngine,
                        moves: Sequence[tuple[str, tuple[int, int]]],
                        *,
                        rules: str = "chinese", komi: float = 7.5,
                        size: int = 9,
                        prob_floor: float = 1e-6) -> dict:
    """计算当前落子方的点位代价表。

    返回 ``{"loss": 9x9, "used_probability": 9x9, "legal_moves": [...],
    "pass_probability": float, "colour": "b"/"w", "turn": int}``。
    """
    result = engine.query_moves(moves, rules=rules, komi=komi, size=size)
    table = result.pointwise_loss(size, prob_floor)
    used = [[0.0] * size for _ in range(size)]
    for (row, col) in result.legal_moves:
        used[row][col] = result.policy[row * size + col]
    return {
        "loss": table,
        "used_probability": used,
        "legal_moves": result.legal_moves,
        "illegal_moves": result.illegal_moves,
        "pass_probability": result.pass_probability,
        "colour": result.colour,
        "turn": result.turn,
        "policy": result.policy,
    }


def loss_from_position(engine: KataGoEngine, history: Sequence[tuple[str, tuple[int, int]]],
                       position: Position, prob_floor: Optional[float] = None) -> dict:
    """给定候选历史与局面元信息，计算动态点位 loss。"""
    return compute_katago_loss(
        engine, history,
        rules=position.rules or "chinese",
        komi=position.komi,
        size=position.size,
        prob_floor=prob_floor if prob_floor is not None else engine.loss_config.prob_floor,
    )
