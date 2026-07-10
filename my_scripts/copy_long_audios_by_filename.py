#!/usr/bin/env python3
"""递归查询指定目录下的音频文件，通过文件名判断时长并复制超过 1 分钟的音频。

示例：
  splited_wavs/6月2日铁前会/00-05-59.250_00-07-01.340.wav
会被复制到:
  out-dir/6月2日铁前会/00-05-59.250_00-07-01.340.wav

只复制文件名中包含起止时间的音频，保持原目录结构不变。
"""

import argparse
import os
import re
import shutil
import sys
from typing import Iterator, Optional, Pattern, Tuple

AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wavpack', '.opus'}

TIMESTAMP_RE = re.compile(
    r'(?P<start>\d{2}[-:]\d{2}[-:]\d{2}(?:\.\d+)?)[_ ](?P<end>\d{2}[-:]\d{2}[-:]\d{2}(?:\.\d+)?)'
)


def parse_timestamp(value: str) -> Optional[float]:
    """Parse HH-MM-SS(.mmm) or HH:MM:SS(.mmm) into seconds."""
    value = value.replace('-', ':')
    parts = value.split(':')
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def parse_duration_from_filename(filename: str) -> Optional[float]:
    """从文件名中解析起止时间并返回持续时长，无法解析时返回 None。"""
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return None
    start = parse_timestamp(match.group('start'))
    end = parse_timestamp(match.group('end'))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def iter_audio_files(root: str, exts: set[str]) -> Iterator[Tuple[str, str]]:
    """遍历根目录下所有音频文件，返回 (绝对路径, 相对路径)。"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)
            yield abs_path, rel_path


def copy_if_long(src: str, rel_path: str, root: str, out_root: str, min_seconds: float) -> bool:
    duration = parse_duration_from_filename(os.path.basename(src))
    if duration is None:
        return False
    if duration <= min_seconds:
        return False
    dst = os.path.join(out_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description='递归复制文件名中持续时长超过指定阈值的音频，保留原目录结构。')
    parser.add_argument('root', help='待扫描的源目录')
    parser.add_argument('out_dir', help='复制到的目标目录')
    parser.add_argument('--min-duration', type=float, default=60.0, help='最小时长阈值，单位秒，默认 60')
    parser.add_argument('--exts', default=','.join(sorted(AUDIO_EXTS)), help='允许处理的音频后缀，逗号分隔，默认包含常见音频格式')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_dir)
    exts = {e if e.startswith('.') else f'.{e}' for e in args.exts.split(',') if e}

    if not os.path.isdir(root):
        print(f'错误：源目录不存在或不是目录：{root}', file=sys.stderr)
        return 1

    copied = 0
    skipped_no_timestamp = 0
    skipped_short = 0

    for src, rel_path in iter_audio_files(root, exts):
        duration = parse_duration_from_filename(os.path.basename(src))
        if duration is None:
            skipped_no_timestamp += 1
            continue
        if duration <= args.min_duration:
            skipped_short += 1
            continue
        dst = os.path.join(out_root, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f'复制: {rel_path} ({duration:.3f}s)')

    print('----------')
    print(f'共复制文件: {copied}')
    print(f'跳过无时间戳文件: {skipped_no_timestamp}')
    print(f'跳过时长不超过 {args.min_duration} 秒的文件: {skipped_short}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
