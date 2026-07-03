#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os
from pathlib import Path
from typing import Optional

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def get_audio_duration(path: Path) -> Optional[float]:
    if sf is None:
        raise RuntimeError("soundfile is required for this script. Install it with `pip install soundfile`.")

    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def resolve_audio_path(audio_value, line_no: int, jsonl_file: Path, root: Optional[Path]) -> Optional[Path]:
    if not isinstance(audio_value, str):
        print(f"跳过第 {line_no} 行：audio 字段不是字符串 -> {audio_value}")
        return None

    audio_path = Path(audio_value)
    if audio_path.is_absolute():
        return audio_path

    if root:
        return root / audio_path

    return jsonl_file.parent / audio_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 JSONL 读取音频路径并统计所有音频总时长。"
    )
    parser.add_argument("jsonl", help="输入 JSONL 文件路径")
    parser.add_argument(
        "--audio-field",
        default="audio",
        help="JSONL 中音频路径字段名，默认 audio",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="如果音频路径是相对路径，指定一个基准目录；默认使用 JSONL 文件所在目录。",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="遇到不存在的音频文件时只警告并继续统计。",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL 文件不存在: {jsonl_path}")

    root_dir = Path(args.root) if args.root else None
    if root_dir is not None and not root_dir.is_dir():
        raise SystemExit(f"指定的根目录不存在: {root_dir}")

    total_duration = 0.0
    total_count = 0
    missing_count = 0
    bad_count = 0
    parse_errors = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过第 {line_no} 行：JSON 解析失败 -> {exc}")
                parse_errors += 1
                continue

            if args.audio_field not in record:
                print(f"跳过第 {line_no} 行：缺少字段 '{args.audio_field}'")
                missing_count += 1
                continue

            audio_file = resolve_audio_path(record[args.audio_field], line_no, jsonl_path, root_dir)
            if audio_file is None:
                missing_count += 1
                continue

            if not audio_file.exists():
                print(f"跳过第 {line_no} 行：文件不存在 -> {audio_file}")
                missing_count += 1
                if args.skip_missing:
                    continue
                else:
                    continue

            duration = get_audio_duration(audio_file)
            if duration is None:
                print(f"无法解析第 {line_no} 行音频时长 -> {audio_file}")
                bad_count += 1
                continue

            total_duration += duration
            total_count += 1

    print("\n统计结果:")
    print(f"JSONL 文件: {jsonl_path}")
    print(f"音频数量: {total_count}")
    print(f"总时长 (秒): {total_duration:.3f}")
    print(f"总时长 (格式化): {format_seconds(total_duration)}")
    print(f"缺失或解析失败: {missing_count + bad_count + parse_errors}")
    print(f"  - JSON 解析错误: {parse_errors}")
    print(f"  - 缺失音频路径或文件: {missing_count}")
    print(f"  - 无法解析时长: {bad_count}")


if __name__ == "__main__":
    main()
