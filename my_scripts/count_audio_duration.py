#!/usr/bin/env python3
"""
递归统计目录下有效音频总时长（秒）。

优先使用内建 wave 解析 WAV 文件；对其他格式尝试调用 `ffprobe` 获取时长（需系统安装 ffprobe）。
"""
import argparse
import contextlib
import os
import subprocess
import wave
from typing import Optional


def get_wav_duration(path: str) -> Optional[float]:
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def get_duration_ffprobe(path: str) -> Optional[float]:
    """Use ffprobe to get the duration in seconds. Returns None on failure."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode != 0:
            return None
        s = res.stdout.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def is_audio_file(name: str, exts: set) -> bool:
    return os.path.splitext(name)[1].lower() in exts


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def walk_and_sum(root: str, exts: set):
    total = 0.0
    count = 0
    bad = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not is_audio_file(fn, exts):
                continue
            path = os.path.join(dirpath, fn)
            dur = None
            ext = os.path.splitext(fn)[1].lower()
            if ext == ".wav":
                dur = get_wav_duration(path)
            else:
                dur = get_duration_ffprobe(path)
            if dur is None:
                bad += 1
            else:
                total += dur
                count += 1
    return total, count, bad


def main():
    parser = argparse.ArgumentParser(description="递归统计目录下有效音频时长（优先 WAV，本地需安装 ffprobe 用于其它格式）")
    parser.add_argument("root", help="要统计的目录路径")
    parser.add_argument(
        "--exts",
        help="以逗号分隔的文件扩展名列表（默认：.wav,.mp3,.m4a,.flac,.aac）",
        default=".wav,.mp3,.m4a,.flac,.aac",
    )
    args = parser.parse_args()

    root = args.root or "/tcdata2/lyb_voice/finetune/splited_wavs/"
    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        raise SystemExit(2)

    exts = set(e if e.startswith(".") else f".{e}" for e in args.exts.split(","))

    total, count, bad = walk_and_sum(root, exts)

    print(f"扫描目录: {root}")
    print(f"匹配扩展: {sorted(exts)}")
    print(f"有效音频文件数: {count}")
    print(f"无法解析/损坏文件数: {bad}")
    print(f"总时长 (秒): {total:.3f}")
    print(f"总时长 (格式化): {format_seconds(total)}")


if __name__ == "__main__":
    main()
