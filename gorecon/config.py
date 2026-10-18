"""配置读取与校验。

默认配置文件位于项目根目录的 ``config.toml``。所有参数都是可修改的示例值。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 回退
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


def resolve_resource(path_value: str) -> str:
    """把配置里的资源路径解析成绝对路径。

    * 绝对路径原样返回；
    * 相对路径按**本项目根目录**解析（例如 "katago/katago.exe"），
      这样整个文件夹可以整体搬走，不依赖任何文件夹外的内容。
    """
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


class ConfigError(ValueError):
    """配置文件缺失或缺键。"""


@dataclass
class EngineConfig:
    katago_path: str
    model_path: str
    analysis_config: str
    startup_timeout_ms: int = 1_800_000
    query_timeout_ms: int = 600_000
    overrides: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)

    @property
    def max_visits(self) -> int:
        return int(self.search.get("maxVisits", 100))

    @property
    def ignore_pre_root_history(self) -> bool:
        return bool(self.search.get("ignorePreRootHistory", False))


@dataclass
class GameConfig:
    board_size: int = 9
    rules: str = "chinese"
    komi: float = 7.5


@dataclass
class SearchConfig:
    """主程序（main/main.py）的按层启发式搜索参数，对应 config.toml 的 [search] 段。"""

    #: 每层全局保留的活动路径数 n（不是每个父节点各留 n 条）。
    beam_width: int = 50
    #: 成功时最多输出多少条路径的 SGF。
    top_k: int = 5
    #: 单条路径手数上限，超过即杀死该路径。
    max_moves: int = 100
    #: 搜索总时间预算（秒）。
    time_budget_s: float = 3600.0
    #: KataGo 查询次数上限。
    engine_query_budget: int = 20000
    #: 排序键里"还差多少目标棋子"的引导权重；0 表示严格只用每步平均 loss。
    progress_weight: float = 1.0
    #: KataGo 项权重 alpha：alpha * (-log p) 计入累计代价。
    alpha_katago: float = 1.0


@dataclass
class TerminalLossConfig:
    """固定终盘点位代价表参数。

    代价档位（以"当前准备落子的颜色"看终局信息）：

    * 终盘为同色棋子 -> ``0``，全局唯一的最小值（每放对一个子就减 1 阶）；
    * 终盘为对方棋子 -> ``opponent_stone_penalty``，一档适中的有限罚分。这一点是搜索
      把棋子放对位置的主要路径（对方这块子被提掉、自己在此落子才可能变成同色），
      因此不能是最贵的一档；
    * 终盘为对方棋子、且该点位于**本方团块内部**（该点相邻的全部在盘内棋子都是同色）
      -> 按位置分三档取更低的罚分：``interior_stone_penalty_corner``（角，2 邻点）、
      ``interior_stone_penalty_edge``（边，3 邻点）、``interior_stone_penalty_standard``
      （常规，4 邻点）。团块内部往往是曾经发生提子的地方，历史上更可能真的放过对方
      的子。三档分开设置是为了以后按位置分别调整（角/边/常规出现对方子的概率不同）；
    * 终盘为空、空区只被对方完全包围 -> 按 ``mode`` 指定的曲线取罚分，默认
      **指数**：``base_temp_penalty * exp(area_coefficient * area) - offset``，
      当前参数即 ``exp(0.12 * area) - 1``（面积 0 时为 0）。``area_coefficient`` 应取
      **小值**，这样小面积区域只是适中、只有面积大了才会明显变贵。
      ``max_penalty`` 是可选截断上限：设为 0 或负数表示**不截断**。也支持
      ``mode = "linear"`` 作为对照；
    * 终盘为空、但空区不是只被对方包围（自己完全包围或中立空区）->
      ``closed_empty_penalty``，一档大的有限罚分（填自己的空、或往中立空区填子都不
      可能是历史）。

    除第一档外都要求"有限"：自己的子也可能在重建历史里被提掉后补回，无限代价会剪掉
    合法路径。
    """

    lambda_total: float = 1.0
    #: 对方包围空区的面积曲线形式：
    #:   "exp"    -> base * exp(k * area) - offset
    #:   "linear" -> base + k * area - offset
    mode: str = "exp"
    #: 曲线前系数。取 1.0（即"不额外乘系数"）。
    base_temp_penalty: float = 1.0
    #: 面积系数 k。指数形式下建议取小值，控制"随面积变贵"的速度。
    area_coefficient: float = 0.12
    #: 曲线整体下移量。取 1.0，使面积 0 时恰好为 0（exp(0) - 1 = 0）。
    offset: float = 1.0
    #: 对方包围空区罚分的可选截断上限。设为 0 或负数表示**不截断**。
    max_penalty: float = 0.0
    #: 终盘为空、但空区不是"只被对方完全包围"（自己完全包围，或中立空区）时的罚分。
    closed_empty_penalty: float = 8.0
    #: 终盘为对方棋子时的适中罚分。
    opponent_stone_penalty: float = 3.0
    #: 以下三档：终盘为对方棋子、且该点位于本方团块内部（相邻的全部在盘内棋子同色）。
    #: 按位置分开设置，便于以后按"角/边/常规出现对方子的概率不同"分别调整。
    #: 这一版三档取同值。
    interior_stone_penalty_corner: float = 1.0
    interior_stone_penalty_edge: float = 1.0
    interior_stone_penalty_standard: float = 1.0


@dataclass
class KataGoLossConfig:
    prob_floor: float = 1e-6


@dataclass
class Config:
    engine: EngineConfig
    game: GameConfig
    search: SearchConfig
    terminal_loss: TerminalLossConfig
    katago_loss: KataGoLossConfig
    path: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.is_file():
            raise ConfigError(f"配置文件不存在: {p}")
        with p.open("rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data, path=str(p))

    @classmethod
    def from_dict(cls, data: dict, path: Optional[str] = None) -> "Config":
        engine_raw = data.get("engine")
        if not engine_raw:
            raise ConfigError("[engine] 段缺失")
        for key in ("katago_path", "model_path", "analysis_config"):
            if not engine_raw.get(key):
                raise ConfigError(f"[engine] 缺少 {key}")
        engine = EngineConfig(
            katago_path=str(engine_raw["katago_path"]),
            model_path=str(engine_raw["model_path"]),
            analysis_config=str(engine_raw["analysis_config"]),
            startup_timeout_ms=int(engine_raw.get("startup_timeout_ms", 1_800_000)),
            query_timeout_ms=int(engine_raw.get("query_timeout_ms", 600_000)),
            overrides=dict(engine_raw.get("overrides", {})),
            search=dict(engine_raw.get("search", {})),
        )
        # 相对路径按项目根目录解析，保证整个文件夹可移动、不依赖外部
        engine.katago_path = resolve_resource(engine.katago_path)
        engine.model_path = resolve_resource(engine.model_path)
        engine.analysis_config = resolve_resource(engine.analysis_config)
        game_raw = dict(data.get("game", {}))
        game = GameConfig(
            board_size=int(game_raw.get("board_size", 9)),
            rules=str(game_raw.get("rules", "chinese")),
            komi=float(game_raw.get("komi", 7.5)),
        )
        search_raw = dict(data.get("search", {}))
        search = SearchConfig(
            beam_width=int(search_raw.get("beam_width", 50)),
            top_k=int(search_raw.get("top_k", 5)),
            max_moves=int(search_raw.get("max_moves", 100)),
            time_budget_s=float(search_raw.get("time_budget_s", 3600.0)),
            engine_query_budget=int(search_raw.get("engine_query_budget", 20000)),
            progress_weight=float(search_raw.get("progress_weight", 1.0)),
            alpha_katago=float(search_raw.get("alpha_katago", 1.0)),
        )
        tl = dict(data.get("terminal_loss", {}))
        terminal = TerminalLossConfig(
            lambda_total=float(tl.get("lambda_total", 1.0)),
            mode=str(tl.get("mode", "exp")),
            base_temp_penalty=float(tl.get("base_temp_penalty", 1.0)),
            area_coefficient=float(tl.get("area_coefficient", 0.12)),
            offset=float(tl.get("offset", 1.0)),
            max_penalty=float(tl.get("max_penalty", 0.0)),
            closed_empty_penalty=float(tl.get("closed_empty_penalty", 8.0)),
            opponent_stone_penalty=float(tl.get("opponent_stone_penalty", 3.0)),
            interior_stone_penalty_corner=float(
                tl.get("interior_stone_penalty_corner", 1.0)),
            interior_stone_penalty_edge=float(tl.get("interior_stone_penalty_edge", 1.0)),
            interior_stone_penalty_standard=float(
                tl.get("interior_stone_penalty_standard", 1.0)),
        )
        kl = dict(data.get("katago_loss", {}))
        katago = KataGoLossConfig(prob_floor=float(kl.get("prob_floor", 1e-6)))
        return cls(engine=engine, game=game, search=search, terminal_loss=terminal,
                   katago_loss=katago, path=path)

    def validate_paths(self) -> None:
        for label, value in (("katago_path", self.engine.katago_path),
                             ("model_path", self.engine.model_path),
                             ("analysis_config", self.engine.analysis_config)):
            if not Path(value).is_file():
                raise ConfigError(f"[engine] {label} 指向的文件不存在: {value}")
