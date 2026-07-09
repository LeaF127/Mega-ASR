#!/usr/bin/env python3
"""
遍历指定目录下的音频文件，检测并移除中间静音段，保留每个非静音片段前后最多 `pad` 秒静音。

实现思路：使用 `ffmpeg` 的 `silencedetect` 分析静音区间，再用 `ffmpeg` `atrim`/`concat` 或 `-af silenceremove` 等手段生成新音频。
本脚本依赖 `ffmpeg`/`ffprobe` 在 PATH 中可用。

python my_scripts/remove_silence_segments.py /data1/lyb_voice/finetune --pad 1.0 --silence-thresh -40 --silence-dur 0.5
"""
import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_silence_intervals(path: str, silence_thresh: float = -40.0, silence_duration: float = 0.5) -> List[Tuple[float, float]]:
    """Return list of (silence_start, silence_end) in seconds using ffmpeg's silencedetect."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={silence_thresh}dB:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]
    proc = run(cmd)
    stderr = proc.stderr
    intervals: List[Tuple[float, float]] = []
    start = None
    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[-1].strip())
            except Exception:
                start = None
        elif "silence_end:" in line:
            try:
                parts = line.split("silence_end:")[-1].strip()
                # format: <end> | silence_duration: <dur>
                end = float(parts.split("|")[0].strip())
                if start is not None:
                    intervals.append((start, end))
                start = None
            except Exception:
                start = None
    return intervals


def invert_intervals(intervals: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    """Given silence intervals, return list of non-silent intervals covering the whole file."""
    if not intervals:
        return [(0.0, total_duration)]
    intervals = sorted(intervals)
    parts: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s, e in intervals:
        if s > prev_end:
            parts.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < total_duration:
        parts.append((prev_end, total_duration))
    return parts


def get_duration_ffprobe(path: str) -> float:
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
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_text = res.stdout.strip()
    if not duration_text or duration_text.upper() == "N/A":
        raise ValueError(f"ffprobe 无法获取时长: {path}")
    try:
        return float(duration_text)
    except ValueError as exc:
        raise ValueError(f"ffprobe 返回了无效时长 {duration_text!r}: {path}") from exc


def generate_trim_commands(non_silent: List[Tuple[float, float]], pad: float, src: str, dst: str) -> List[str]:
    """Generate ffmpeg filter_complex commands to trim and concat non-silent parts while keeping up to `pad` seconds around each part."""
    segs = []
    for start, end in non_silent:
        s = max(0.0, start - pad)
        e = end + pad
        segs.append((s, e))
    # Create a filter that trims segments and concatenates them
    filters = []
    inputs = []
    for i, (s, e) in enumerate(segs):
        filters.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
        inputs.append(f"[a{i}]")
    filter_complex = ";".join(filters) + ";" + "".join(inputs) + f"concat=n={len(segs)}:v=0:a=1[out]"
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        dst,
    ]
    return cmd


def process_file(path: str, out_path: str, pad: float, silence_thresh: float, silence_dur: float):
    try:
        total = get_duration_ffprobe(path)
        intervals = ffprobe_silence_intervals(path, silence_thresh=silence_thresh, silence_duration=silence_dur)
        non_silent = invert_intervals(intervals, total)
        # Filter out very short non-silent pieces
        non_silent = [p for p in non_silent if p[1] - p[0] > 0.01]
        if not non_silent:
            # nothing to keep, copy original
            subprocess.run(["cp", path, out_path])
            return True
        cmd = generate_trim_commands(non_silent, pad, path, out_path)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(proc.stderr.strip())
        return proc.returncode == 0
    except Exception as exc:
        print(f"处理异常: {path} -> {exc}")
        return False


def print_progress(index: int, total: int) -> None:
    if total <= 0:
        return
    width = 30
    filled = int(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = index / total * 100
    sys.stdout.write(f"\r进度: [{bar}] {index}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="移除音频中间静音并保留片段前后最多 pad 秒静音（依赖 ffmpeg/ffprobe）")
    parser.add_argument("root", help="要处理的目录（会递归处理音频文件）")
    parser.add_argument("--out-dir", help="输出目录（默认在 root 下创建 trimmed_audio）", default=None)
    parser.add_argument("--pad", type=float, default=1.0, help="每段前后保留的静音时长（秒）")
    parser.add_argument("--silence-thresh", type=float, default=-40.0, help="静音阈值，dBFS，默认 -40")
    parser.add_argument("--silence-dur", type=float, default=0.5, help="判定静音的最短时长（秒）")
    parser.add_argument("--exts", default=".wav,.mp3,.m4a,.flac,.aac")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_dir or os.path.join(root, "trimmed_audio"))
    exts = set(e if e.startswith(".") else f".{e}" for e in args.exts.split(","))
    os.makedirs(out_root, exist_ok=True)

    audio_files: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Avoid walking into the output tree if it is inside the source root.
        dirnames[:] = [d for d in dirnames if not os.path.commonpath([os.path.abspath(os.path.join(dirpath, d)), out_root]) == out_root]
        rel = os.path.relpath(dirpath, root)
        target_dir = os.path.join(out_root, rel) if rel != "." else out_root
        os.makedirs(target_dir, exist_ok=True)
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(target_dir, fn)
            if os.path.exists(dst):
                print(f"跳过（输出已存在）: {src} -> {dst}")
                continue
            audio_files.append((src, dst))

    total = len(audio_files)
    print(f"共发现 {total} 个待处理音频文件")
    for index, (src, dst) in enumerate(audio_files, 1):
        print_progress(index, total)
        print(f"\n[{index}/{total}] 处理: {src} -> {dst}")
        try:
            ok = process_file(src, dst, pad=args.pad, silence_thresh=args.silence_thresh, silence_dur=args.silence_dur)
        except Exception as exc:
            print(f"处理异常: {src} -> {exc}")
            ok = False
        if not ok:
            print(f"处理失败，复制原文件: {src}")
            subprocess.run(["cp", src, dst])

    print("\n处理完成")


if __name__ == "__main__":
    main()
