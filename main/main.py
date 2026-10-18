"""主程序：从终盘局面推断能合法重放到该局面的棋谱。

原始要求（本文件原有注释，保留在此）：

    根据 go_construction_instruction_for_agents.md 的指导；
    利用 prior 下已验证的两个 loss 成果；
    每层保留前 50 条"每步平均 loss"最小的路径（不足则保留尽量多），
    输出前 5 条 loss 最小的路径，各存一个 SGF 到 prediction/ 目录；
    终止约束：与原始棋谱的终盘状态相同（逐点一致）；超过 100 步自动杀死该路径；
    若无法到达终盘状态，不输出棋谱，只用文字提示。

对应实现：

  * 两个 loss：
      - gorecon.terminal_loss  固定终盘点位 loss，**只由终局生成一次**，整次搜索不变；
      - gorecon.katago_loss    动态 KataGo 点位 loss，每手从**落子前**的父局面查询；
  * 单步累计代价（第 6 节）：
      C(H⊕(c,x)) = C(H) - log p_K(x|H,c) + lambda * L_终盘^c(x)
  * 搜索（第 7 节 + 上面的要求）：每层把所有父候选展开出的合法子节点合并成**一个池**，
    按**每步平均 loss**（= 总代价 / 已下手数）从小到大统一保留 n=50 条；
    n 是全局每层上限，不是每个父节点各留 n 条。

所有可调参数都在项目根的 config.toml（[search] / [engine] / [engine.search] / [game] /
[terminal_loss]），本文件不含隐藏数值。

用法（最简单）::

    直接改下面 INPUT_SGF / CONFIG_TOML / OUTPUT_DIR 三个绝对路径，然后运行：

        python main/main.py

命令行参数都是**可选**的覆盖项，不写就用文件顶部的路径 + config.toml 的值::

    python main/main.py --input  <终盘局面.sgf>
    python main/main.py --sgf    <完整棋谱.sgf>
    python main/main.py --beam-width 50 --top-k 5 --max-moves 100 --alpha-katago 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# =============================================================================
#  ★★★ 输入路径：待输入 ★★★
# =============================================================================

# 默认输入：仓库自带示例棋谱（manual/ 里那盘 9 路对局）的**终局棋谱**，
# 由 prior/from_manual_to_situation.py 生成在 situation/ 下（扁平结构，无子目录）。
# 想换一盘，用命令行：  python main/main.py --input <终局棋谱或完整棋谱>
# 留空（""）时运行程序会提示输入；两种文件都支持（完整棋谱会自动重放取终盘）。
# 相对路径按本文件夹解析；绝对路径也可。
INPUT_SGF = r"situation\KaTrain_人类 (分先对局) vs 人类 (分先对局) 2026-10-15 21 06 00.sgf"

# 参数文件（相对本文件夹；一般不用改）
CONFIG_TOML = "config.toml"

# 输出目录（相对本文件夹；成功时把前 top_k 条候选的 SGF 与 loss 明细写到这里）
OUTPUT_DIR = "prediction"

# =============================================================================
#  直接运行：  python main/main.py
#  （命令行参数都是可选的覆盖项，不写就用上面的默认值 + config.toml）
# =============================================================================

# 让 main/ 下的脚本能 import 项目根的 gorecon
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gorecon.board import BLACK, point_to_sgf
from gorecon.config import Config, ConfigError
from gorecon.console import enable_utf8
from gorecon.katago_loss import KataGoEngine, KataGoError
from gorecon.search import SearchConfig, heuristic_search, verify_path
from gorecon.sgf_io import (SgfError, load_game_record, load_position, position_from_record,
                            write_moves)
from gorecon.terminal_loss import build_terminal_loss

def _as_project_path(value: str) -> str:
    """相对路径按本文件夹解析成绝对路径；绝对路径原样返回。"""
    p = Path(value)
    return str(p) if p.is_absolute() else str((PROJECT_ROOT / p).resolve())


DEFAULT_CONFIG = _as_project_path(CONFIG_TOML)
DEFAULT_OUT_DIR = _as_project_path(OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="由终盘局面正向启发式搜索重建棋谱（每层保留每步平均 loss 最小的路径）")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", default=None,
                     help=f"终盘局面 SGF（只含摆子）；默认用文件顶部的 INPUT_SGF")
    src.add_argument("--sgf", default=None,
                     help=f"完整棋谱 SGF；默认用文件顶部的 INPUT_SGF")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="成功路径 SGF 的输出目录（失败时不写任何文件）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--beam-width", type=int, default=None,
                        help="每层全局保留的路径数 n（覆盖 config 的 [search] beam_width）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="成功时输出前 k 条路径（覆盖 config 的 top_k）")
    parser.add_argument("--max-moves", type=int, default=None,
                        help="单条路径手数上限，超过即杀（覆盖 config 的 max_moves）")
    parser.add_argument("--time-budget", type=float, default=None, help="搜索时间预算（秒）")
    parser.add_argument("--query-budget", type=int, default=None, help="KataGo 查询次数上限")
    parser.add_argument("--progress-weight", type=float, default=None,
                        help="排序中'还差多少目标棋子'的引导权重；0 表示严格只用每步平均 loss")
    parser.add_argument("--alpha-katago", type=float, default=None,
                        help="KataGo 项权重 alpha（覆盖 config 的 [search] alpha_katago）；1 为原始定义")
    parser.add_argument("--max-visits", type=int, default=None, help="覆盖引擎 maxVisits")
    parser.add_argument("--no-engine", action="store_true",
                        help="不调用 KataGo（只用于逻辑自检，代价里没有 KataGo 项）")
    parser.add_argument("--log", default=None, help="把搜索日志另存到该文件")
    parser.add_argument("--verbose", action="store_true", help="打印引擎日志")
    return parser


def print_active_parameters(config: Config, args: argparse.Namespace, search: SearchConfig) -> None:
    e, g, tc = config.engine, config.game, config.terminal_loss
    print("=" * 76)
    print(f"参数来源：{config.path}")
    print("  [engine]")
    print(f"    katago_path       = {e.katago_path}")
    print(f"    model_path        = {e.model_path}")
    print(f"    analysis_config   = {e.analysis_config}")
    print(f"    overrides         = {e.overrides}")
    print(f"    maxVisits         = {e.max_visits}"
          f"{'  <- 已由命令行覆盖' if args.max_visits is not None else ''}")
    print(f"    ignorePreRootHistory = {e.ignore_pre_root_history}")
    print("  [game]")
    print(f"    board_size={g.board_size}  rules={g.rules}  komi={g.komi}")
    print("  [search]")
    print(f"    beam_width          = {search.beam_width}   （每层全局保留的路径数）")
    print(f"    top_k               = {search.top_k}   （成功时输出前几条）")
    print(f"    max_moves           = {search.max_moves}   （超过即杀该路径）")
    print(f"    time_budget_s       = {search.time_budget_s:g}")
    print(f"    engine_query_budget = {search.engine_query_budget}")
    print(f"    progress_weight     = {search.progress_weight:g}   （0 = 严格只用每步平均 loss）")
    print(f"    alpha_katago        = {search.alpha_katago:g}   "
          f"（KataGo 项权重；1 = 说明里的原始定义）")
    print("  [terminal_loss]（固定终盘点位 loss，只由终局生成一次）")
    print(f"    lambda_total={tc.lambda_total:g}  mode={tc.mode!r}  base={tc.base_temp_penalty:g}  "
          f"k={tc.area_coefficient:g}  offset={tc.offset:g}  max_penalty={tc.max_penalty:g}")
    print(f"    closed_empty={tc.closed_empty_penalty:g}  "
          f"opponent_stone={tc.opponent_stone_penalty:g}  "
          f"interior(角/边/常规)={tc.interior_stone_penalty_corner:g}/"
          f"{tc.interior_stone_penalty_edge:g}/{tc.interior_stone_penalty_standard:g}")
    print(f"  输出目录：{args.out_dir}")
    print("=" * 76)


def ask_input_path() -> str:
    """输入路径待输入时，交互式询问用户。"""
    print("=" * 70)
    print("没有设置输入路径（main/main.py 顶部 INPUT_SGF 为空，也没用命令行参数）。")
    print("请输入要重建的棋谱/局面 SGF 路径：可直接把文件拖进终端；直接回车退出。")
    try:
        text = input("> ")
    except EOFError:
        text = ""
    return text.strip().strip('"').strip("'")


def resolve_input_path(given: str) -> Path:
    """把用户给的路径变成绝对路径：去掉拖拽产生的引号，相对路径按本文件夹解析。"""
    text = str(given).strip().strip('"').strip("'")
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_target(args: argparse.Namespace, config: Config):
    """取得目标终盘局面，返回 (position, source_label, 输入文件路径)。

    输入路径优先用命令行，其次用文件顶部的 INPUT_SGF；两者都为空时交互式询问。
    该文件是完整棋谱还是只含终盘的摆子局面，自动判断。
    """
    given = args.sgf or args.input or INPUT_SGF
    if not given:
        given = ask_input_path()
        if not given:
            raise SgfError("没有提供输入文件，已退出。")

    path = resolve_input_path(given)
    if not path.is_file():
        raise SgfError(f"输入文件不存在：{path}")

    if args.sgf:
        mode = "record"
    elif args.input:
        mode = "position"
    else:
        # 自动判断：含 B/W 着手属性的是棋谱，否则是摆子局面
        head = path.read_text(encoding="utf-8", errors="replace")[:4096]
        mode = "record" if (";B[" in head or ";W[" in head) else "position"

    if mode == "record":
        record = load_game_record(path)
        if not record.moves:
            raise SgfError(f"{path.name}: 棋谱没有落子")
        position = position_from_record(record)   # 整局重放后的终盘
        return position, f"{path.name} 的终盘（重放 {len(record.moves)} 手）", path

    position = load_position(path,
                             expected_size=config.game.board_size,
                             expected_komi=config.game.komi,
                             expected_rules=config.game.rules)
    return position, str(path), path


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    args = build_parser().parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    if args.max_visits is not None:
        config.engine.search["maxVisits"] = args.max_visits

    try:
        position, source, input_path = load_target(args, config)
    except SgfError as exc:
        print(f"[输入错误] {exc}", file=sys.stderr)
        return 2

    search = SearchConfig(
        beam_width=args.beam_width if args.beam_width is not None else config.search.beam_width,
        top_k=args.top_k if args.top_k is not None else config.search.top_k,
        max_moves=args.max_moves if args.max_moves is not None else config.search.max_moves,
        time_budget_s=(args.time_budget if args.time_budget is not None
                       else config.search.time_budget_s),
        engine_query_budget=(args.query_budget if args.query_budget is not None
                             else config.search.engine_query_budget),
        progress_weight=(args.progress_weight if args.progress_weight is not None
                         else config.search.progress_weight),
        alpha_katago=(args.alpha_katago if args.alpha_katago is not None
                      else config.search.alpha_katago),
        lambda_terminal=config.terminal_loss.lambda_total,
    )
    print_active_parameters(config, args, search)

    target = position.board
    black_stones, white_stones = target.stone_count(BLACK), target.stone_count("w")
    print(f"目标终盘：{source}")
    print(f"  {position.size} 路，{black_stones} 黑 / {white_stones} 白，"
          f"规则 {position.rules}，贴目 {position.komi}，"
          f"合计 {black_stones + white_stones} 个棋子，"
          f"首次出现空点的手数约为 {black_stones + white_stones} 手之后")

    # 固定终盘点位 loss：只由终局生成一次，整次搜索不变
    terminal = build_terminal_loss(target, config.terminal_loss)
    print(f"  固定终盘 loss 表已生成（固定空区 {len(terminal.regions)} 个，整次搜索不变）")

    engine = None
    if not args.no_engine:
        engine = KataGoEngine(config.engine, config.katago_loss, verbose=args.verbose)
        try:
            t0 = time.time()
            engine.start()
        except KataGoError as exc:
            print(f"[引擎错误] {exc}", file=sys.stderr)
            return 3
        print(f"KataGo 就绪（启动 {time.time() - t0:.2f}s，maxVisits={config.engine.max_visits}）")

    print("\n开始搜索：")
    started = time.time()
    try:
        outcome = heuristic_search(position, terminal, engine, search, target=target)
    finally:
        if engine is not None:
            engine.close()
    elapsed = time.time() - started

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(outcome.log) + "\n", encoding="utf-8")
        print(f"\n搜索日志 -> {log_path}")

    print("\n" + "=" * 76)
    print(f"搜索结束：{outcome.stop_reason}")
    print(f"  层数={outcome.layers}  展开节点={outcome.expanded}  "
          f"引擎查询={outcome.engine_queries}  被杀路径={outcome.killed_over_max_moves}  "
          f"用时={elapsed:.1f}s")

    if not outcome.found:
        print("\n未找到能与终盘逐点一致的棋谱，因此不输出任何 SGF。")
        print("可尝试的手段：调大 config.toml 中 [search] 的 beam_width / max_moves / "
              "time_budget_s / engine_query_budget，或调整 [terminal_loss] 的权重。")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch not in '\\/:*?"<>| ' else "_"
                   for ch in input_path.stem)

    print(f"\n找到 {len(outcome.paths)} 条与终盘逐点一致的路径，"
          f"按每步平均 loss 从小到大输出：")
    records = []
    for rank, candidate in enumerate(outcome.paths, start=1):
        check = verify_path(candidate, target)
        if not check["ok"]:  # pragma: no cover - 说明实现有 bug
            print(f"  [内部错误] 第 {rank} 条未通过独立重放校验：{check['reason']}",
                  file=sys.stderr)
            return 4
        comment = (f"重建候选 第{rank}名；手数={candidate.steps} "
                   f"每步平均loss={candidate.average_loss:.4f} "
                   f"总loss={candidate.total_cost:.4f} "
                   f"(KataGo={candidate.cumulative_katago:.4f}, "
                   f"终盘={candidate.cumulative_terminal:.4f})")
        path = write_moves(out_dir / f"{stem}_candidate{rank}.sgf", list(candidate.moves),
                           size=position.size, komi=position.komi, rules=position.rules,
                           comment=comment)
        # 每条候选另存一份 loss 明细：总量 + 逐手分项
        per_move = []
        running = 0.0
        for i, ((colour, point), kg, tl) in enumerate(zip(
                candidate.moves, candidate.step_katago, candidate.step_terminal), start=1):
            kg_w = search.alpha_katago * kg
            tl_w = search.lambda_terminal * tl
            running += kg_w + tl_w
            per_move.append({
                "turn": i,
                "move": f"{colour.upper()}{point_to_sgf(*point)}",
                "katago_loss": kg,
                "katago_loss_weighted": kg_w,
                "terminal_loss": tl,
                "terminal_loss_weighted": tl_w,
                "step_total": kg_w + tl_w,
                "cumulative_total": running,
            })
        loss_path = out_dir / f"{stem}_candidate{rank}_loss.json"
        loss_path.write_text(json.dumps({
            "rank": rank,
            "sgf": str(path),
            "moves": candidate.steps,
            "weights": {
                "alpha_katago": search.alpha_katago,
                "lambda_terminal": search.lambda_terminal,
                "note": "单步代价 = alpha_katago * (-log p_K) + lambda_terminal * L_终盘",
            },
            "totals": {
                "katago_loss": candidate.cumulative_katago,
                "terminal_loss": candidate.cumulative_terminal,
                "total_loss": candidate.total_cost,
                "average_loss_per_move": candidate.average_loss,
                "katago_loss_unweighted": candidate.raw_katago_sum,
                "terminal_loss_unweighted": candidate.raw_terminal_sum,
            },
            "per_move": per_move,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  第{rank}名：{candidate.steps} 手  每步平均loss={candidate.average_loss:.4f}  "
              f"总loss={candidate.total_cost:.4f}"
              f"(KataGo={candidate.cumulative_katago:.4f}, "
              f"终盘={candidate.cumulative_terminal:.4f})")
        print(f"          SGF  -> {path}")
        print(f"          loss -> {loss_path}")
        records.append({
            "rank": rank,
            "file": str(path),
            "loss_file": str(loss_path),
            "moves": candidate.steps,
            "average_loss": candidate.average_loss,
            "total_loss": candidate.total_cost,
            "katago_loss": candidate.cumulative_katago,
            "terminal_loss": candidate.cumulative_terminal,
            "moves_sgf": [f"{c.upper()}{point_to_sgf(*p)}" for c, p in candidate.moves],
        })

    summary = out_dir / f"{stem}_summary.json"
    summary.write_text(json.dumps({
        "target": source,
        "size": position.size,
        "komi": position.komi,
        "rules": position.rules,
        "target_black_stones": black_stones,
        "target_white_stones": white_stones,
        "search": {
            "beam_width": search.beam_width,
            "top_k": search.top_k,
            "max_moves": search.max_moves,
            "progress_weight": search.progress_weight,
            "lambda_terminal": search.lambda_terminal,
        },
        "outcome": {
            "stop_reason": outcome.stop_reason,
            "layers": outcome.layers,
            "expanded": outcome.expanded,
            "engine_queries": outcome.engine_queries,
            "killed_over_max_moves": outcome.killed_over_max_moves,
            "elapsed_s": elapsed,
        },
        "paths": records,
        "note": "每条路径都经过独立重放校验，终盘逐点一致。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
