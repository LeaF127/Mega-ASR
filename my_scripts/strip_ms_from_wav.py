#!/usr/bin/env python3
"""
递归处理目录下所有 .wav 文件，将文件名中的毫秒部分删除，只保留到秒。

命名规则: HH-MM-SS.mmm_HH-MM-SS.mmm.wav → HH-MM-SS_HH-MM-SS.wav

用法:
    python my_scripts/strip_ms_from_wav.py --dir /path/to/audio
    python my_scripts/strip_ms_from_wav.py --dir /path/to/audio --dry-run
"""

import argparse
import re
from pathlib import Path

# 匹配 HH-MM-SS.mmm (或 HH:MM:SS.mmm) 格式，提取时分秒，丢弃毫秒
TIMESTAMP_MS_RE = re.compile(r'(\d{1,2})[-:](\d{2})[-:](\d{2})\.\d+')


def strip_milliseconds(filename: str) -> str | None:
    """去除文件名中所有时间戳的毫秒部分，返回新文件名；无需修改则返回 None。"""
    new_name = TIMESTAMP_MS_RE.sub(r'\1-\2-\3', filename)
    if new_name == filename:
        return None
    return new_name


def main():
    parser = argparse.ArgumentParser(
        description='递归去除 .wav 文件名中的毫秒部分'
    )
    parser.add_argument(
        '--dir', required=True,
        help='要处理的根目录'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='仅预览，不实际重命名'
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}")
        return

    wav_files = sorted(root.rglob('*.wav'))
    if not wav_files:
        print("未找到任何 .wav 文件")
        return

    renamed = 0
    skipped = 0

    for fpath in wav_files:
        old_name = fpath.name
        new_name = strip_milliseconds(old_name)
        if new_name is None:
            skipped += 1
            continue

        new_path = fpath.with_name(new_name)
        if new_path.exists():
            print(f"跳过 (目标已存在): {fpath}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY RUN] {old_name} → {new_name}")
        else:
            fpath.rename(new_path)
            print(f"重命名: {old_name} → {new_name}")
        renamed += 1

    print(f"\n完成: 重命名 {renamed} 个, 跳过 {skipped} 个, 共 {len(wav_files)} 个 .wav 文件")
    if args.dry_run:
        print("(这是预览模式，未实际修改任何文件)")


if __name__ == '__main__':
    main()
