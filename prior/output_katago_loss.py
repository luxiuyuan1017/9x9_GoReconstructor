"""任务二：输入棋谱与第 n 手，输出 KataGo 在该手处的损失值，形式为 9x9 热图（一张）。

================================ 参数在哪 ================================
所有可调参数都在项目根的 **config.toml**，本文件不含隐藏的默认值：

  [engine]          katago_path        KataGo 可执行程序
                    model_path         冻结模型权重
                    analysis_config    官方分析配置（内含 OpenCL 调优设置）
                    startup_timeout_ms 首次查询等待上限（OpenCL 首轮自动调优可能数分钟）
                    query_timeout_ms   单条查询等待上限
                    overrides          额外传给引擎的覆盖项（numAnalysisThreads、
                                       numSearchThreads、nnMaxBatchSize、nnRandomize 等）
  [engine.search]   maxVisits          每手查询的搜索数（用户指定 100）
                    ignorePreRootHistory 必须为 false，保留候选历史对预测的影响
  [game]            board_size / rules / komi    棋盘尺寸、规则、贴目
  [katago_loss]     prob_floor         合法点概率下限，避免 -log(0)

（本任务不使用 [terminal_loss] 段，那是任务三的参数。）

=========================================================================

约定与来源（项目说明第 5、6 节）：

* 第 ``n`` 手（1 基）的损失值由**落子前**的父局面给出 —— 每手的 KataGo 分数必须从
  落子前取得，不能在落子后查询已经被占据的位置；
* 使用本地冻结 KataGo 的**原始 policy**，每手查询的搜索数取 ``maxVisits``；
* 点位损失 = ``-log p(x)``；非法点的 policy 为 0，不取对数，热图中标 ``×``；
* pass 是 policy 最后一项，不在 9x9 热图中展开。

输出：一个 PNG 热图 + 一个 JSON（含损失表、原始 policy、实战手概率与名次）。

用法::

    python prior/output_katago_loss.py --sgf "manual/xxx.sgf" --move 25
    python prior/output_katago_loss.py --sgf "manual/xxx.sgf" --move 25 --metric regret
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gorecon.board import BLACK, point_to_sgf
from gorecon.config import Config, ConfigError
from gorecon.console import enable_utf8
from gorecon.heatmap import draw_heatmap, grid_as_text
from gorecon.katago_loss import KataGoEngine, KataGoError
from gorecon.sgf_io import SgfError, load_game_record

# ---------------------------------------------------------------------------
# 本文件里的常量：只有"参数文件在哪"和"输出到哪"，不含任何可调数值
# 所有 loss / 引擎参数都在 config.toml 里
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"    # 参数都从这里读
DEFAULT_OUT_DIR = PROJECT_ROOT / "output"        # 热图与 JSON 输出目录

METRIC_LABELS = {
    "nll": "-log p(x)  (nats)",
    "regret": "log p(x)/p(played)  (nats)",
    "policy": "raw policy probability",
}


def print_active_parameters(config: Config, args: argparse.Namespace, sgf_path: Path) -> None:
    """打印本次运行实际生效的参数及其出处，便于定位该改 config.toml 的哪一项。"""
    e, g, k = config.engine, config.game, config.katago_loss
    print("=" * 74)
    print(f"参数来源：{config.path}")
    print("  [engine]")
    print(f"    katago_path        = {e.katago_path}")
    print(f"    model_path         = {e.model_path}")
    print(f"    analysis_config    = {e.analysis_config}")
    print(f"    overrides          = {e.overrides}")
    print(f"    startup_timeout_ms = {e.startup_timeout_ms}")
    print(f"    query_timeout_ms   = {e.query_timeout_ms}")
    print("  [engine.search]")
    print(f"    maxVisits          = {e.max_visits}"
          f"{'   <- 本次由命令行 --max-visits 覆盖' if args.max_visits is not None else ''}")
    print(f"    ignorePreRootHistory = {e.ignore_pre_root_history}")
    print("  [game]")
    print(f"    board_size={g.board_size}  rules={g.rules}  komi={g.komi}")
    print("  [katago_loss]")
    print(f"    prob_floor         = {k.prob_floor}")
    print("  本次运行")
    print(f"    sgf={sgf_path.name}  move={args.move}  metric={args.metric}"
          f"  out_dir={args.out_dir}")
    print("=" * 74)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="某手处的 KataGo 点位损失（9x9 热图）")
    parser.add_argument("--sgf", required=True, help="输入棋谱 SGF（含 B/W 着手）")
    parser.add_argument("--move", type=int, required=True, help="第 n 手（1 基）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-name", default=None, help="输出文件名前缀（默认 <棋谱名>_move<NNN>）")
    parser.add_argument("--metric", choices=tuple(METRIC_LABELS), default="nll",
                        help="热图呈现的量；nll 与 regret 同量纲，policy 为原始概率")
    parser.add_argument("--max-visits", type=int, default=None, help="覆盖 maxVisits")
    parser.add_argument("--no-plot", action="store_true", help="只写 JSON，不画图")
    parser.add_argument("--verbose", action="store_true")
    return parser


def analyse_move(engine: KataGoEngine, record, move_number: int, size: int,
                 komi: float, rules: str, prob_floor: float) -> dict:
    """计算第 ``move_number`` 手（1 基）处的损失信息。"""
    if not 1 <= move_number <= len(record.moves):
        raise SgfError(f"--move 必须在 1..{len(record.moves)} 之间，收到 {move_number}")

    colour, played = record.moves[move_number - 1]
    history = list(record.moves[: move_number - 1])
    result = engine.query(history, turn=len(history), rules=rules, komi=komi, size=size,
                          colour=colour)

    nll = [[math.inf] * size for _ in range(size)]
    regret = [[math.inf] * size for _ in range(size)]
    policy_grid = [[0.0] * size for _ in range(size)]
    p_played = max(result.policy[played[0] * size + played[1]], prob_floor)
    played_nll = -math.log(p_played)

    for (row, col) in result.legal_moves:
        p = max(result.policy[row * size + col], prob_floor)
        nll[row][col] = -math.log(p)
        regret[row][col] = math.log(p / p_played)
        policy_grid[row][col] = result.policy[row * size + col]

    ranked = sorted(result.legal_moves, key=lambda pt: -result.policy[pt[0] * size + pt[1]])
    rank = ranked.index(played) + 1 if played in ranked else None
    best = ranked[0]
    p_best = result.policy[best[0] * size + best[1]]

    return {
        "move_number": move_number,
        "colour": colour,
        "played_sgf": point_to_sgf(*played),
        "played_probability": result.policy[played[0] * size + played[1]],
        "played_nll": played_nll,
        "played_regret_vs_best": math.log(max(p_best, prob_floor) / p_played),
        "policy_rank": rank,
        "legal_count": len(result.legal_moves),
        "policy_top1": {"sgf": point_to_sgf(*best), "probability": p_best},
        "pass_probability": result.pass_probability,
        "illegal_points": [list(pt) for pt in result.illegal_moves],
        "nll": nll,
        "regret": regret,
        "policy": policy_grid,
        "top10_policy": [
            {"sgf": point_to_sgf(*pt), "probability": result.policy[pt[0] * size + pt[1]]}
            for pt in ranked[:10]
        ],
    }


def _safe_stem(stem: str) -> str:
    for ch in '\\/:*?"<>| ':
        stem = stem.replace(ch, "_")
    return stem


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    args = build_parser().parse_args(argv)
    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    try:
        record = load_game_record(args.sgf)
    except SgfError as exc:
        print(f"[输入错误] {exc}", file=sys.stderr)
        return 2
    if args.max_visits is not None:
        config.engine.search["maxVisits"] = args.max_visits

    size, komi, rules = record.size, record.komi, record.rules
    stem = args.out_name or _safe_stem(f"{Path(args.sgf).stem}_move{args.move:03d}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_active_parameters(config, args, Path(args.sgf))
    print(f"棋谱 {Path(args.sgf).name}：{len(record.moves)} 手，{size} 路，"
          f"规则 {rules}，贴目 {komi}")

    engine = KataGoEngine(config.engine, config.katago_loss, verbose=args.verbose)
    try:
        engine.start()
    except KataGoError as exc:
        print(f"[引擎错误] {exc}", file=sys.stderr)
        return 3
    try:
        # 预热：吸收首次查询的固定开销，使该手耗时可读
        t0 = time.time()
        engine.query([], turn=0, rules=rules, komi=komi, size=size, colour=BLACK)
        print(f"引擎就绪（预热 {time.time() - t0:.2f}s）")
        t1 = time.time()
        info = analyse_move(engine, record, args.move, size, komi, rules,
                            config.katago_loss.prob_floor)
        info["engine_query_seconds"] = time.time() - t1
    finally:
        engine.close()

    grid = {"nll": info["nll"], "regret": info["regret"], "policy": info["policy"]}[args.metric]
    print(f"\n--- 第 {args.move} 手：{info['colour'].upper()}{info['played_sgf']} "
          f"（查询 {info['engine_query_seconds']:.2f}s）---")
    print(grid_as_text(grid))
    print(f"实战手 {info['colour'].upper()}{info['played_sgf']}："
          f"p={info['played_probability']:.5f} nll={info['played_nll']:.4f} "
          f"名次 {info['policy_rank']}/{info['legal_count']} "
          f"相对一选 {info['played_regret_vs_best']:.4f}")
    print(f"一选 {info['policy_top1']['sgf']} p={info['policy_top1']['probability']:.5f}；"
          f"pass p={info['pass_probability']:.5f}")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    print(f"JSON -> {json_path}")

    if not args.no_plot:
        png_path = out_dir / f"{stem}.png"
        # 不叠画棋子：棋子会挡住格内的损失值；也不画标题（长文件名会被裁切）
        draw_heatmap(grid, png_path,
                     cmap="viridis" if args.metric == "policy" else "inferno",
                     cbar_label=METRIC_LABELS[args.metric])
        print(f"PNG  -> {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
