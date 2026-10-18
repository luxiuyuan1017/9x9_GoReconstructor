"""任务三：输入**终局局面**，输出重建损失值（9x9 热图，黑白各一张）。

================================ 参数在哪 ================================
所有可调参数都在项目根的 **config.toml**，本文件不含隐藏的默认值。
本任务只用到两段：

  [game]            board_size / rules / komi
                    棋盘尺寸、规则、贴目（读取局面 SGF 时会与之校验）

  [terminal_loss]   重建损失的固定终盘项，逐项含义：
    lambda_total                    整体系数 lambda（总代价 = KataGo 项 + lambda × 终盘项）
    mode                            对方包围空区的曲线形式 "exp" / "linear"
    base_temp_penalty               曲线基准系数（exp 时是面积 0 的基准值）
    area_coefficient                面积系数 k，控制随面积变贵的速度
    max_penalty                     对方包围空区罚分的硬上限
    closed_empty_penalty            自己完全包围 / 中立空区的罚分
    opponent_stone_penalty          终盘为对方棋子（普通点）的罚分
    interior_stone_penalty_corner   对方棋子、且在本方团块内部（角，2 邻点）
    interior_stone_penalty_edge     同上（边，3 邻点）
    interior_stone_penalty_standard 同上（常规，4 邻点）

（本任务不使用 [engine] / [katago_loss] 段，那是任务二的参数。）

=========================================================================

重建损失的第一项、也是唯一只依赖终局信息的一项，是**固定终盘点位代价表**
（项目说明第 4 节）：由终局局面在整张 9x9 棋盘上计算一次，之后整次搜索都不再改变。

关键约束（这正是本任务的设计前提）：

* 它**只能使用终盘信息**。终盘是唯一已知的局面，中间任何一手的盘面都不可获取，
  因此这里**不接受**"第 n 手"之类的中间局面输入 —— 那样等于用了不该有的信息；
* 代价表按颜色分两张（黑方落子视角、白方落子视角），同一终局只算一次；
* 动态的 KataGo 项 ``-log p`` 依赖候选历史，无法由终局推出，不属于本任务输出。

固定空区定义（项目说明第 4 节）：在终局中用上下左右连接划分完整空区；棋盘边缘只充当
边界，不是一种棋子颜色；只接触黑棋的空区视为被黑方包围，只接触白棋的视为被白方包围；
接触双方或没有接触棋子的空区不判为任何一方完全包围。

输出：两张 PNG（黑方 / 白方落子视角）+ 一份 JSON（含两张代价表、空区明细与分类）。

用法::

    python prior/output_reconstruction_loss.py --situation situation/xxx/final.sgf
    python prior/output_reconstruction_loss.py --sgf "manual/xxx.sgf"     # 自动取终盘
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gorecon.board import BLACK, WHITE
from gorecon.config import Config, ConfigError
from gorecon.console import enable_utf8
from gorecon.heatmap import draw_heatmap, grid_as_text
from gorecon.sgf_io import SgfError, load_game_record, load_position, position_from_record
from gorecon.terminal_loss import build_terminal_loss, classify

# ---------------------------------------------------------------------------
# 本文件里的常量：只有"参数文件在哪"和"输出到哪"，不含任何可调数值
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"    # 参数都从这里读
DEFAULT_OUT_DIR = PROJECT_ROOT / "output"        # 热图与 JSON 输出目录


def print_active_parameters(config: Config, args: argparse.Namespace, lam: float) -> None:
    """打印本次运行实际生效的参数及其出处，便于定位该改 config.toml 的哪一项。"""
    g, tc = config.game, config.terminal_loss
    print("=" * 74)
    print(f"参数来源：{config.path}")
    print("  [game]")
    print(f"    board_size = {g.board_size}   rules = {g.rules}   komi = {g.komi}")
    print("  [terminal_loss]")
    print(f"    lambda_total                    = {tc.lambda_total}"
          f"{'   <- 本次由命令行 --lambda-terminal 覆盖为 ' + format(lam, 'g') if args.lambda_terminal is not None else ''}")
    print(f"    mode                            = {tc.mode!r}"
          f"   （对方包围空区的曲线形式）")
    print(f"    base_temp_penalty               = {tc.base_temp_penalty:g}")
    print(f"    area_coefficient                = {tc.area_coefficient:g}")
    print(f"    max_penalty                     = {tc.max_penalty:g}")
    print(f"    closed_empty_penalty            = {tc.closed_empty_penalty:g}"
          f"   （自己完全包围 / 中立空区）")
    print(f"    opponent_stone_penalty          = {tc.opponent_stone_penalty:g}"
          f"   （终盘为对方棋子）")
    print(f"    interior_stone_penalty_corner   = {tc.interior_stone_penalty_corner:g}"
          f"   （团块内部·角）")
    print(f"    interior_stone_penalty_edge     = {tc.interior_stone_penalty_edge:g}"
          f"   （团块内部·边）")
    print(f"    interior_stone_penalty_standard = {tc.interior_stone_penalty_standard:g}"
          f"   （团块内部·常规）")
    print("  本次运行")
    print(f"    out_dir={args.out_dir}")
    print("=" * 74)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="由终局局面生成固定终盘点位重建损失（9x9 热图，黑白各一张）")
    parser.add_argument("--situation", default=None, help="终局局面 SGF（只含摆子）")
    parser.add_argument("--sgf", default=None, help="棋谱 SGF；自动重放后取终盘作为终局局面")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-name", default=None, help="输出文件名前缀")
    parser.add_argument("--lambda-terminal", type=float, default=None,
                        help="覆盖终盘项整体系数 lambda；设为 0 可得到全 0 的对照表")
    parser.add_argument("--no-plot", action="store_true", help="只写 JSON，不画图")
    return parser


def _safe_stem(stem: str) -> str:
    for ch in '\\/:*?"<>| ':
        stem = stem.replace(ch, "_")
    return stem


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    args = build_parser().parse_args(argv)
    if bool(args.situation) == bool(args.sgf):
        print("[输入错误] 必须且只能给出 --situation 或 --sgf 之一", file=sys.stderr)
        return 2

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    # ---- 取得唯一的已知局面：终局 ----
    if args.sgf:
        try:
            record = load_game_record(args.sgf)
        except SgfError as exc:
            print(f"[输入错误] {exc}", file=sys.stderr)
            return 2
        position = position_from_record(record)   # 整局重放后的终盘
        source = f"{Path(args.sgf).name} 的终盘（重放 {len(record.moves)} 手）"
    else:
        try:
            position = load_position(args.situation,
                                     expected_size=config.game.board_size,
                                     expected_komi=config.game.komi,
                                     expected_rules=config.game.rules)
        except SgfError as exc:
            print(f"[输入错误] {exc}", file=sys.stderr)
            return 2
        source = str(args.situation)

    lam = (args.lambda_terminal if args.lambda_terminal is not None
           else config.terminal_loss.lambda_total)
    target = position.board
    table = build_terminal_loss(target, config.terminal_loss)

    black_stones, white_stones = target.stone_count(BLACK), target.stone_count(WHITE)
    stem = args.out_name or _safe_stem(
        Path(args.sgf or args.situation).stem + "_terminal_loss")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_active_parameters(config, args, lam)
    print(f"终局局面：{source}")
    print(f"  {position.size} 路，{black_stones} 黑 / {white_stones} 白，"
          f"规则 {position.rules}，贴目 {position.komi}")
    print(f"  固定空区 {len(table.regions)} 个（整次搜索只算一次）：")
    for region in table.regions:
        owner = ("黑方包围" if region["enclosed_by_black"] else
                 "白方包围" if region["enclosed_by_white"] else "无单一包围方")
        print(f"    空区 {region['id']}: 面积 {region['area']}，{owner}，"
              f"黑方落子罚 {region['black_penalty']:.2f}，白方落子罚 {region['white_penalty']:.2f}")

    payload = {
        "source": source,
        "size": position.size,
        "komi": position.komi,
        "rules": position.rules,
        "player_to_move": position.player_to_move,
        "black_stones": black_stones,
        "white_stones": white_stones,
        "lambda_terminal": lam,
        "terminal_config": vars(config.terminal_loss),
        "terminal_loss_black": table.black_map,
        "terminal_loss_white": table.white_map,
        "region_id": table.region_id,
        "regions": table.regions,
        "classification_black": [[classify(table, target, BLACK, r, c)
                                  for c in range(position.size)]
                                 for r in range(position.size)],
        "classification_white": [[classify(table, target, WHITE, r, c)
                                  for c in range(position.size)]
                                 for r in range(position.size)],
        "note": "重建损失的终盘项只能由终局局面生成，整次搜索只算一次，与任何中间盘面无关。"
                "它按颜色分两张表，故输出两张热图。",
    }

    for colour in (BLACK, WHITE):
        grid = table.map_for(colour)
        print(f"\n--- 固定终盘点位重建损失（{colour.upper()} 方落子视角；"
              f"{colour} 在终局同色点=0）---")
        print(grid_as_text(grid))
        if not args.no_plot:
            png = out_dir / f"{stem}_{colour}.png"
            # 不叠画棋子：棋子会挡住格内的损失值
            draw_heatmap(grid, png, cmap="inferno", cbar_label="fixed terminal loss")
            print(f"PNG  -> {png}")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    print(f"\nJSON -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
