"""任务一：读取 manual/ 中的真实棋谱，重放后生成**只含终盘**的标准化局面 SGF。

按项目说明第 11 节的数据制作要求：输入必须由完整真实棋谱**重放**后新建只含终盘的
摆子 SGF，不能直接拿棋谱当输入（读取端也会拒绝含 B/W 着手的文件）。

用法示例::

    python prior/from_manual_to_situation.py
    python prior/from_manual_to_situation.py --input manual/xxx.sgf
    python prior/from_manual_to_situation.py --save-original   # 额外另存原始顺序备用
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
from gorecon.console import enable_utf8
from gorecon.sgf_io import SgfError, load_game_record, position_from_record, write_moves, write_position

DEFAULT_MANUAL_DIR = PROJECT_ROOT / "manual"
DEFAULT_SITUATION_DIR = PROJECT_ROOT / "situation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实棋谱 -> 只含终盘的标准化局面 SGF")
    parser.add_argument("--manual-dir", default=str(DEFAULT_MANUAL_DIR), help="棋谱目录")
    parser.add_argument("--out-dir", default=str(DEFAULT_SITUATION_DIR), help="局面输出目录")
    parser.add_argument("--input", nargs="*", default=None,
                        help="指定棋谱文件（默认处理 --manual-dir 下全部 .sgf）")
    parser.add_argument("--pattern", default="*.sgf", help="棋谱文件匹配模式")
    parser.add_argument("--save-original", action="store_true",
                        help="额外把原始顺序棋谱另存到 _original/，仅供评测（默认不生成）")
    parser.add_argument("--komi", type=float, default=None, help="覆盖贴目（默认取棋谱中的 KM）")
    parser.add_argument("--rules", default=None, help="覆盖规则名（默认取棋谱中的 RU）")
    return parser


def find_game_files(args: argparse.Namespace) -> list[Path]:
    if args.input:
        files = [Path(p) for p in args.input]
    else:
        files = sorted(Path(args.manual_dir).glob(args.pattern))
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SgfError("以下棋谱文件不存在：\n  " + "\n  ".join(missing))
    if not files:
        raise SgfError(f"在 {args.manual_dir} 中找不到匹配 {args.pattern} 的棋谱")
    return files


def convert_one(path: Path, args: argparse.Namespace) -> dict:
    record = load_game_record(path)
    if args.komi is not None:
        record.komi = args.komi
    if args.rules is not None:
        record.rules = args.rules
    if not record.moves:
        raise SgfError(f"{path.name}: 棋谱没有落子，无法生成终盘局面")

    # 从空盘重新重放整局，确保生成的是"重放结果"而不是原棋谱的摆子
    board_final, captured_all = record.replay()
    total_captured = sum(len(c) for c in captured_all)

    name = path.stem
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    comment = (f"来源 {path.name} 第 {len(record.moves)} 手后（终盘）；"
               f"原棋谱共 {len(record.moves)} 手，累计被提 {total_captured} 子")
    target = out_dir / "final.sgf"
    write_position(target, position_from_record(record), comment=comment)

    written = [str(target)]
    if args.save_original:
        original_dir = Path(args.out_dir) / "_original"
        original_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_dir / f"{name}.sgf"
        write_moves(original_path, list(record.moves), size=record.size, komi=record.komi,
                    rules=record.rules,
                    comment=f"原始顺序副本，仅供评测，不得作为搜索输入。来源 {path.name}")
        written.append(str(original_path))

    return {
        "source": str(path),
        "name": name,
        "size": record.size,
        "komi": record.komi,
        "rules": record.rules,
        "moves": len(record.moves),
        "player_to_move_at_final": (WHITE if record.moves[-1][0] == BLACK else BLACK),
        "final_black_stones": board_final.stone_count(BLACK),
        "final_white_stones": board_final.stone_count(WHITE),
        "captured_total": total_captured,
        "outputs": written,
    }


def main(argv: list[str] | None = None) -> int:
    enable_utf8()
    args = build_parser().parse_args(argv)
    try:
        files = find_game_files(args)
    except SgfError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for path in files:
        try:
            info = convert_one(path, args)
        except SgfError as exc:
            print(f"[跳过] {path.name}: {exc}", file=sys.stderr)
            continue
        records.append(info)
        print(f"{path.name}: {info['moves']} 手 -> 终盘 {info['final_black_stones']} 黑 / "
              f"{info['final_white_stones']} 白，累计被提 {info['captured_total']} 子，"
              f"轮到 {'黑' if info['player_to_move_at_final'] == BLACK else '白'}方；"
              f"写出 {info['outputs'][0]}")

    if not records:
        print("没有任何棋谱被成功转换。", file=sys.stderr)
        return 1

    index_path = out_root / "index.json"
    payload = {
        "note": "由 prior/from_manual_to_situation.py 生成；每份棋谱输出一个只含终盘的摆子 SGF。",
        "position_point_order": "row-major, row 0 = top, col 0 = left",
        "games": records,
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共转换 {len(records)} 份棋谱；索引写入 {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
