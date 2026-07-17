#!/usr/bin/env python3
"""
递归遍历目录，从文件名计算音频时长，筛选超过指定时长的 wav 文件并输出路径。

文件名格式: HH-MM-SS[.mmm]_HH-MM-SS[.mmm].wav
时长 = 结束时间戳 - 起始时间戳

用法:
    python my_scripts/find_long_audios.py --dir /path/to/audio --out long_audios.txt
    python my_scripts/find_long_audios.py --dir /path/to/audio --out long_audios.txt --min-duration 60
"""

import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 正则：匹配 HH-MM-SS[.mmm]_HH-MM-SS[.mmm].wav
# ---------------------------------------------------------------------------
SEGMENT_FILENAME_RE = re.compile(
    r"^(\d{2}-\d{2}-\d{2})(?:\.\d{3})?_(\d{2}-\d{2}-\d{2})(?:\.\d{3})?\.wav$"
)


def parse_segment_timestamp(timestamp_text: str) -> float:
    """将 HH-MM-SS 格式的时间戳转换为秒数。"""
    parts = timestamp_text.split("-")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def get_duration_from_filename(filename: str) -> float | None:
    """从文件名计算音频时长（秒），无法解析则返回 None。"""
    match = SEGMENT_FILENAME_RE.match(filename)
    if not match:
        return None
    start_sec = parse_segment_timestamp(match.group(1))
    end_sec = parse_segment_timestamp(match.group(2))
    return end_sec - start_sec


def main():
    parser = argparse.ArgumentParser(
        description="从文件名计算时长，筛选超过指定阈值的 wav 文件。"
    )
    parser.add_argument(
        "--dir", required=True,
        help="要搜索的根目录（递归查找 *.wav）",
    )
    parser.add_argument(
        "--out", required=True,
        help="输出文件路径，每行一个音频绝对路径",
    )
    parser.add_argument(
        "--min-duration", type=float, default=60.0,
        help="最小时长阈值（秒），默认 60（1 分钟）",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}")
        return

    wav_files = sorted(root.rglob("*.wav"))
    if not wav_files:
        print("未找到任何 .wav 文件")
        return

    long_files = []
    skipped = 0

    for wav_path in wav_files:
        duration = get_duration_from_filename(wav_path.name)
        if duration is None:
            skipped += 1
            continue
        if duration > args.min_duration:
            long_files.append(wav_path.resolve().as_posix())

    # 写入输出文件
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(long_files) + "\n", encoding="utf-8")

    print(
        f"完成: 共扫描 {len(wav_files)} 个 wav, "
        f"超过 {args.min_duration}s 的有 {len(long_files)} 个, "
        f"无法解析文件名 {skipped} 个"
    )
    print(f"结果已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
