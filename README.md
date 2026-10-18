# 9x9 KataReconstructor — 自包含发布版

[中文](#中文) · [English](#english)

---

<a id="中文"></a>
## 中文

9 路围棋**终盘棋谱重建**原型。输入一个只含终盘的 9×9 局面（或一盘完整棋谱），
从空盘开始按黑白交替做**正向按层启发式搜索**，输出能合法重放到该局面的棋谱。

本目录是**自包含**的：KataGo 可执行文件、依赖 DLL、官方 analysis 配置、
OpenCL 调优缓存和模型权重都在目录内，不依赖文件夹外的任何内容。

### 目录结构

```
9x9_KataReconstructor_commit/
├─ main/main.py                  主程序：按层启发式搜索重建棋谱
├─ prior/                        四个可独立运行的子任务
│   ├─ from_manual_to_situation.py   完整棋谱 -> 只含终盘的摆子局面 SGF
│   ├─ output_katago_loss.py         某一手的 KataGo loss 9x9 热图
│   ├─ output_reconstruction_loss.py 终盘的固定重建损失热图（黑白各一张）
│   └─ katago_loss_per_move.py       逐手 KataGo loss 明细
├─ docs/
│   ├─ go_construction_instruction_for_agents.md   需求与设计说明（原始）
│   └─ NOTES_pitfalls.md                           开发中踩过的坑与易错点
├─ gorecon/                      共享库
│   ├─ board.py                  棋盘、合法性、提子、空区连通、坐标适配
│   ├─ sgf_io.py                 SGF 读写与严格校验
│   ├─ terminal_loss.py          固定终盘点位 loss
│   ├─ katago_loss.py            常驻 KataGo analysis 客户端
│   ├─ search.py                 按层启发式搜索
│   ├─ heatmap.py                9x9 热图输出
│   ├─ config.py                 config.toml 读取（相对路径按本目录解析）
│   └─ console.py                Windows 控制台 UTF-8 保护
├─ katago/
│   ├─ katago.exe                KataGo 1.18.1（OpenCL 版）
│   ├─ *.dll                     运行所需 DLL
│   ├─ analysis_config.cfg       官方 analysis 配置
│   └─ KataGoData/opencltuning/  已附 OpenCL 调优缓存（免首次自动调优）
├─ model/                        冻结模型 g170e-b15c192（9 路可用）
├─ manual/                       示例棋谱（可替换成你自己的）
├─ prediction/                   主程序输出（成功才写 SGF）
├─ output/                       prior 脚本的热图与 JSON
├─ situation/                    由棋谱生成的终盘局面 SGF
├─ config.toml                   全部可调参数（路径均为相对本文件夹）
├─ requirements.txt              Python 依赖
└─ README.md                     本文件（中英双语）
```

### 环境要求

* Python 3.11+（3.11 起自带 `tomllib`；3.10 及以下需装 `tomli`）
* Python 包：见 `requirements.txt`

```bash
pip install -r requirements.txt
```

本机已有配置好的 conda 环境 `go`，可直接用：

```bash
conda run -n go python main/main.py
```

### 快速开始

**1) 重建一盘棋（主程序）**

```bash
python main/main.py
```

`main/main.py` 顶部的 `INPUT_SGF` 是**待输入**的（默认空字符串），运行后会提示：

```
请输入要重建的棋谱/局面 SGF 路径：可直接把文件拖进终端；直接回车退出。
>
```

把 SGF 文件拖进终端（或粘贴路径）回车即可。也可直接用命令行给路径：

```bash
python main/main.py --input manual/你的棋谱.sgf      # 完整棋谱或终盘局面都行
python main/main.py --sgf   manual/你的棋谱.sgf      # 明确按"完整棋谱"读取
```

完整棋谱会自动重放取终盘；只含 `AB`/`AW` 的摆子 SGF 则直接作为目标，程序自动判断。

成功时把前 `top_k` 条候选写到 `prediction/`：

```
prediction/<棋谱名>_candidate1.sgf
prediction/<棋谱名>_candidate1_loss.json     # 该条的 loss 分项（逐手）
prediction/<棋谱名>_candidate2.sgf
...
prediction/<棋谱名>_summary.json             # 本次搜索的汇总
```

若预算内无法到达终盘，**不会写出任何 SGF**，只打印文字提示。

**2) 终盘的重建损失热图（黑白各一张）**

```bash
python prior/output_reconstruction_loss.py --situation situation/<棋谱名>/final.sgf
```

**3) 某一手的 KataGo loss 热图**

```bash
python prior/output_katago_loss.py --sgf manual/你的棋谱.sgf --move 25
```

**4) 逐手 KataGo loss**

```bash
python prior/katago_loss_per_move.py --sgf manual/你的棋谱.sgf
```

**5) 完整棋谱 -> 只含终盘的摆子局面**

```bash
python prior/from_manual_to_situation.py            # 处理 manual/ 下全部 .sgf
```

### 参数（`config.toml`）

脚本启动时会打印本次实际生效的值。

| 段 | 内容 |
| --- | --- |
| `[engine]` | KataGo 路径、模型、analysis 配置（**相对本文件夹**）、超时、`overrides` |
| `[engine.search]` | `maxVisits`（搜索数）、`ignorePreRootHistory` |
| `[game]` | 棋盘尺寸、规则、贴目 |
| `[search]` | `beam_width`(50)、`top_k`、`max_moves`(100)、时间/查询预算、`progress_weight`、`alpha_katago` |
| `[terminal_loss]` | 固定终盘重建损失的各个档位与面积曲线 |
| `[katago_loss]` | 合法点概率下限 |

常用命令行覆盖项（`main/main.py`）：

```
--beam-width 50 --top-k 5 --max-moves 100
--time-budget 3600 --query-budget 20000
--alpha-katago 1 --progress-weight 1 --max-visits 100
```

### 耗时与硬件

* 单次 KataGo 查询约 1 秒（Intel UHD 核显，`maxVisits=100`）。
* 一盘约 50 手的重建，`beam_width=50` 时约 **20~25 分钟**（每层约 50 次查询）。
* 本目录已附带 `katago/KataGoData/opencltuning/`，换机器或显卡时可能需要重新自动调优
  （约 10 分钟，之后写缓存）。

### 说明与限制

* 终盘通常**不能唯一确定**原始历史：输出是搜索预算内找到的高可信候选，
  不保证与真实棋谱逐手相同。
* 固定终盘 loss 只由**终局**生成一次（中间盘面不可获取），与任何中间局面无关。
* 目标函数对"同一批落点的先后次序"不敏感，因此中后盘常出现手顺重排的等价解。
* 目前按 9 路实现；13 路需要调整 `board_size`、坐标字母表与热图坐标轴。

---

<a id="english"></a>
## English

A prototype for **reconstructing the move history** of a 9×9 Go game from its final
position. Given a terminal 9×9 board (or a complete game record), it runs a forward,
layer-by-layer heuristic search from an empty board and outputs game records that legally
replay to that position.

This directory is **self-contained**: the KataGo binary, its DLLs, the official analysis
config, the OpenCL tuning cache and the model weights all live here. Nothing outside this
folder is required.

### Layout

```
9x9_KataReconstructor_commit/
├─ main/main.py                  Entry point: beam-search reconstruction
├─ prior/                        Four independent sub-tasks
│   ├─ from_manual_to_situation.py   Game record -> terminal-only setup SGF
│   ├─ output_katago_loss.py         KataGo loss 9x9 heatmap at one move
│   ├─ output_reconstruction_loss.py Fixed terminal loss heatmaps (B / W)
│   └─ katago_loss_per_move.py       Per-move KataGo loss report
├─ docs/
│   ├─ go_construction_instruction_for_agents.md   Original design spec
│   └─ NOTES_pitfalls.md                           Development pitfalls
├─ gorecon/                      Shared library
│   ├─ board.py                  Board, legality, captures, empty regions, coordinates
│   ├─ sgf_io.py                 SGF I/O with strict validation
│   ├─ terminal_loss.py          Fixed terminal loss
│   ├─ katago_loss.py            Persistent KataGo analysis client
│   ├─ search.py                 Layer-by-layer heuristic search
│   ├─ heatmap.py                9x9 heatmap rendering
│   ├─ config.py                 config.toml loader (relative paths resolved here)
│   └─ console.py                UTF-8 console shim for Windows
├─ katago/
│   ├─ katago.exe                KataGo 1.18.1 (OpenCL)
│   ├─ *.dll                    Required DLLs
│   ├─ analysis_config.cfg       Official analysis config
│   └─ KataGoData/opencltuning/  Bundled OpenCL tuning cache (no re-tune)
├─ model/                        Frozen model g170e-b15c192 (works on 9×9)
├─ manual/                       Sample record (replace with your own)
├─ prediction/                   main.py outputs (SGFs, only on success)
├─ output/                       Heatmaps and JSON from prior scripts
├─ situation/                    Terminal setup SGFs
├─ config.toml                   All tunables (paths relative to this folder)
├─ requirements.txt              Python dependencies
└─ README.md                     This file (Chinese + English)
```

### Requirements

* Python 3.11+ (`tomllib` is in the stdlib; on 3.10 or older install `tomli`)
* Python packages: see `requirements.txt`

```bash
pip install -r requirements.txt
```

A ready conda env `go` exists on this machine:

```bash
conda run -n go python main/main.py
```

### Quick start

**1) Reconstruct a game (main program)**

```bash
python main/main.py
```

`INPUT_SGF` at the top of `main/main.py` is **pending input** (empty by default); the
program prompts for a path:

```
请输入要重建的棋谱/局面 SGF 路径：可直接把文件拖进终端；直接回车退出。
>
```

Drag an SGF file into the terminal (or paste a path) and press Enter. You can also pass
the path directly:

```bash
python main/main.py --input manual/your_game.sgf      # record or setup SGF
python main/main.py --sgf   manual/your_game.sgf      # force "complete record" mode
```

Complete records are replayed to obtain the terminal position; setup-only SGFs
(`AB`/`AW`) are used directly. The mode is detected automatically.

On success the top `top_k` candidates are written to `prediction/`:

```
prediction/<name>_candidate1.sgf
prediction/<name>_candidate1_loss.json     # loss breakdown (per move)
prediction/<name>_candidate2.sgf
...
prediction/<name>_summary.json             # search summary
```

If no candidate reaches the terminal position within budget, **no SGF is written**; only a
text message is printed.

**2) Fixed terminal reconstruction loss (one heatmap per colour)**

```bash
python prior/output_reconstruction_loss.py --situation situation/<name>/final.sgf
```

**3) KataGo loss heatmap for one move**

```bash
python prior/output_katago_loss.py --sgf manual/your_game.sgf --move 25
```

**4) Per-move KataGo loss**

```bash
python prior/katago_loss_per_move.py --sgf manual/your_game.sgf
```

**5) Complete record -> terminal-only setup SGF**

```bash
python prior/from_manual_to_situation.py     # all .sgf under manual/
```

### Parameters (`config.toml`)

Scripts print the parameters actually in effect at start-up.

| Section | Contents |
| --- | --- |
| `[engine]` | KataGo path, model, analysis config (**relative to this folder**), timeouts, `overrides` |
| `[engine.search]` | `maxVisits`, `ignorePreRootHistory` |
| `[game]` | Board size, rules, komi |
| `[search]` | `beam_width` (50), `top_k`, `max_moves` (100), budgets, `progress_weight`, `alpha_katago` |
| `[terminal_loss]` | Bands and area curve of the fixed terminal loss |
| `[katago_loss]` | Probability floor |

Frequently used CLI overrides (`main/main.py`):

```
--beam-width 50 --top-k 5 --max-moves 100
--time-budget 3600 --query-budget 20000
--alpha-katago 1 --progress-weight 1 --max-visits 100
```

### Runtime and hardware

* One KataGo query takes ~1 s (Intel UHD iGPU, `maxVisits=100`).
* A ~50-move reconstruction takes **20–25 minutes** at `beam_width=50` (~50 queries per
  layer).
* The OpenCL tuning cache is bundled under `katago/KataGoData/opencltuning/`. On a
  different machine or GPU, KataGo may need to re-tune once (~10 minutes, then cached).

### Limitations

* The terminal position usually does **not** uniquely determine the history. Outputs are
  high-confidence candidates within budget, not guaranteed exact reconstructions.
* The fixed terminal loss is computed **once from the terminal position** (intermediate
  boards are unavailable).
* The objective ignores move order, so equivalent reorderings appear in the middle and
  endgame.
* Implemented for 9×9; 13×13 requires changing `board_size`, the coordinate alphabet and
  the heatmap axis labels.

### Documentation

* `docs/go_construction_instruction_for_agents.md` — original specification
* `docs/NOTES_pitfalls.md` — pitfalls (coordinates, captures, engine, loss modelling)
