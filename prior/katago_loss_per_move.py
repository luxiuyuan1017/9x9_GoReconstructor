"""逐手统计真实棋谱每一步的 KataGo loss。

口径与任务二一致：第 n 手的 loss 由**落子前**的父局面给出，
取冻结 KataGo 的原始 policy，loss = -log p(实战手)。

输出：
  * output/<棋谱名>_katago_loss_per_move.json  逐手明细 + 汇总
  * 终端打印一张可读的表格
"""

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
from gorecon.katago_loss import KataGoEngine, KataGoError
from gorecon.sgf_io import SgfError, load_game_record

DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output"


def main(argv=None) -> int:
    enable_utf8()
    parser = argparse.ArgumentParser(description="真实棋谱每步的 KataGo loss")
    parser.add_argument("--sgf", required=True, help="输入棋谱 SGF")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-visits", type=int, default=None, help="覆盖 maxVisits")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

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
    floor = config.katago_loss.prob_floor
    print(f"棋谱 {Path(args.sgf).name}：{len(record.moves)} 手，{size} 路，"
          f"规则 {rules}，贴目 {komi}，maxVisits={config.engine.max_visits}")

    engine = KataGoEngine(config.engine, config.katago_loss, verbose=args.verbose)
    try:
        engine.start()
    except KataGoError as exc:
        print(f"[引擎错误] {exc}", file=sys.stderr)
        return 3

    rows = []
    try:
        t0 = time.time()
        engine.query([], turn=0, rules=rules, komi=komi, size=size, colour=BLACK)
        print(f"引擎就绪（预热 {time.time() - t0:.2f}s）\n")
        for i, (colour, point) in enumerate(record.moves, start=1):
            history = list(record.moves[: i - 1])
            t1 = time.time()
            result = engine.query(history, turn=len(history), rules=rules, komi=komi,
                                  size=size, colour=colour)
            p_played = result.policy[point[0] * size + point[1]]
            ranked = sorted(result.legal_moves,
                            key=lambda pt: -result.policy[pt[0] * size + pt[1]])
            rank = ranked.index(point) + 1 if point in ranked else None
            p_best = result.policy[ranked[0][0] * size + ranked[0][1]]
            rows.append({
                "turn": i,
                "move": f"{colour.upper()}{point_to_sgf(*point)}",
                "colour": colour,
                "probability": p_played,
                "katago_loss": -math.log(max(p_played, floor)),
                "rank": rank,
                "legal_count": len(result.legal_moves),
                "top1_move": f"{colour.upper()}{point_to_sgf(*ranked[0])}" if False else point_to_sgf(*ranked[0]),
                "top1_probability": p_best,
                "regret_vs_top1": math.log(max(p_best, floor) / max(p_played, floor)),
                "pass_probability": result.pass_probability,
                "query_seconds": time.time() - t1,
            })
            print(f"  {i:>2}/{len(record.moves)}  {rows[-1]['move']:<5} "
                  f"p={p_played:.5f} loss={rows[-1]['katago_loss']:.4f} "
                  f"名次={rank}/{len(result.legal_moves)}", flush=True)
    finally:
        engine.close()

    losses = [r["katago_loss"] for r in rows]
    ranks = [r["rank"] for r in rows]
    summary = {
        "sgf": str(args.sgf),
        "moves": len(rows),
        "size": size,
        "komi": komi,
        "rules": rules,
        "maxVisits": config.engine.max_visits,
        "model": config.engine.model_path,
        "mean_katago_loss": sum(losses) / len(losses),
        "median_katago_loss": sorted(losses)[len(losses) // 2],
        "min": min(losses), "max": max(losses),
        "mean_rank": sum(ranks) / len(ranks),
        "top1_hits": sum(1 for r in ranks if r == 1),
        "top3_hits": sum(1 for r in ranks if r <= 3),
        "top5_hits": sum(1 for r in ranks if r <= 5),
        "per_move": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(args.sgf).stem.replace(" ", "_") + "_katago_loss_per_move.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("逐手 KataGo loss 汇总（loss = -log p(实战手)）")
    print("=" * 66)
    print(f"  手数 {len(rows)}   平均 {summary['mean_katago_loss']:.4f}   "
          f"中位 {summary['median_katago_loss']:.4f}   "
          f"最小 {summary['min']:.4f}   最大 {summary['max']:.4f}")
    print(f"  实战手平均名次 {summary['mean_rank']:.2f}；"
          f"一选命中 {summary['top1_hits']} 手，前三 {summary['top3_hits']} 手，"
          f"前五 {summary['top5_hits']} 手")
    print(f"\n  {'手数':>4} {'落点':<5} {'概率':>9} {'loss':>8} {'名次':>7} {'一选':<5}")
    for r in rows:
        print(f"  {r['turn']:>4} {r['move']:<5} {r['probability']:>9.5f} "
              f"{r['katago_loss']:>8.4f} {str(r['rank']) + '/' + str(r['legal_count']):>7} "
              f"{r['top1_move']:<5}")
    print(f"\nJSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
