"""控制台输出保护：Windows 控制台默认编码（如 cp932/GBK）无法编码中文日志。

所有命令行入口在打印任何内容之前调用 :func:`enable_utf8`。
"""

from __future__ import annotations

import io
import sys

_configured = False


def enable_utf8() -> None:
    """尽量把 stdout/stderr 切到 UTF-8，并让无法编码的字符降级而不是抛异常。"""
    global _configured
    if _configured:
        return
    _configured = True
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    wrapper = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace",
                                               line_buffering=True)
                    if stream is sys.stdout:
                        sys.stdout = wrapper
                    else:
                        sys.stderr = wrapper
            except Exception:  # pragma: no cover - 极端环境
                pass
