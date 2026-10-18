"""本项目共享的最小库：只服务 prior/ 下的三个任务。

============================  ==========================================
:mod:`board`                  棋盘、合法性、提子、空区连通与坐标适配
:mod:`sgf_io`                 局面/棋谱读写与严格校验
:mod:`terminal_loss`          固定终盘点位 loss（重建损失的终盘项）
:mod:`katago_loss`            动态 KataGo 点位 loss（常驻 analysis 进程客户端）
:mod:`heatmap`                9x9 热图输出
:mod:`config`                 config.toml 读取与校验
:mod:`console`                Windows 控制台 UTF-8 输出保护
============================  ==========================================
"""

from .board import (BLACK, WHITE, Board, IllegalMove, board_from_points, gtp_to_point,
                    point_to_gtp, point_to_sgf, sgf_to_point)
from .config import Config, ConfigError
from .katago_loss import KataGoEngine, KataGoError, KataGoResult, compute_katago_loss
from .sgf_io import (GameRecord, Position, SgfError, load_game_record, load_position,
                     position_from_record, write_moves, write_position)
from .terminal_loss import TerminalLossTable, build_terminal_loss

__all__ = [
    "BLACK", "WHITE", "Board", "IllegalMove", "board_from_points",
    "gtp_to_point", "point_to_gtp", "point_to_sgf", "sgf_to_point",
    "Config", "ConfigError",
    "KataGoEngine", "KataGoError", "KataGoResult", "compute_katago_loss",
    "GameRecord", "Position", "SgfError", "load_game_record", "load_position",
    "position_from_record", "write_moves", "write_position",
    "TerminalLossTable", "build_terminal_loss",
]
