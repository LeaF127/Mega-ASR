#!/usr/bin/env python3
"""
根据 transcript.txt 和切分后的 wav 文件生成训练/验证 JSONL。

- 文本文件 (transcript.txt): 每行格式为 "HH:MM:SS[.mmm] 文本内容"
- 音频文件: 命名格式为 "HH-MM-SS[.mmm]_HH-MM-SS[.mmm].wav"（毫秒可选）
- 通过起始时间戳将音频片段与文本行匹配

用法:
    python my_scripts/build_jsonl_from_transcripts.py \
        --audio-dir /path/to/splited_wavs \
        --text-dir /path/to/keep_folders \
        --output-dir outputs
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import librosa

# ---------------------------------------------------------------------------
# 正则表达式：解析 transcript.txt 中的时间戳行
# 格式: [Speaker 1: ]HH:MM:SS[.mmm] 文本内容
# ---------------------------------------------------------------------------
TIMESTAMP_TEXT_RE = re.compile(
    r'^\s*(?:Speaker\s+\d+\s*[:,]?\s*)?(\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)(?:\s+(.+))?$',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 正则表达式：解析 wav 文件名中的起止时间戳
# 格式: HH-MM-SS[.mmm]_HH-MM-SS[.mmm].wav（毫秒部分可选，适配 strip_ms 后的文件名）
# ---------------------------------------------------------------------------
SEGMENT_FILENAME_RE = re.compile(
    r"^(\d{2}-\d{2}-\d{2})(?:\.\d{3})?_(\d{2}-\d{2}-\d{2})(?:\.\d{3})?\.wav$"
)


def parse_timestamp(timestamp_text: str) -> float:
    """将 HH:MM:SS[.mmm] 格式的时间戳转换为秒数（float）。"""
    normalized = timestamp_text.replace("，", ",").replace(",", ".")
    parts = normalized.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp format: {timestamp_text}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_segment_timestamp(seconds: float) -> str:
    """将秒数转换为 HH-MM-SS.mmm 格式字符串（用于精确匹配）。"""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{ms:03d}"


def format_segment_timestamp_no_ms(seconds: float) -> str:
    """将秒数转换为 HH-MM-SS 格式字符串（无毫秒，用于 strip 后的文件名匹配）。"""
    total_sec = int(round(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def parse_segment_filename(filename: str):
    """从 wav 文件名中解析起止时间戳，返回 (start_sec, end_sec) 或 None。"""
    match = SEGMENT_FILENAME_RE.match(filename)
    if not match:
        return None
    start = match.group(1)  # HH-MM-SS
    end = match.group(2)    # HH-MM-SS
    return parse_segment_timestamp(start), parse_segment_timestamp(end)


def parse_segment_timestamp(timestamp_text: str) -> float:
    """将 HH-MM-SS 格式的片段时间戳转换为秒数。"""
    normalized = timestamp_text.replace("-", ":")
    return parse_timestamp(normalized)


def parse_transcript_file(transcript_path: Path):
    """
    解析 transcript.txt 文件，返回 [(start_sec, text), ...] 列表。

    支持两种格式：
    1. 单行: "HH:MM:SS.mmm 文本内容"
    2. 多行: 时间戳行后跟若干行纯文本，直到遇到下一个时间戳行或空行分隔
    """
    entries = []
    lines = transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        match = TIMESTAMP_TEXT_RE.match(line)
        if not match:
            i += 1
            continue
        start_text = match.group(1)
        text = match.group(2) or ""

        # 如果时间戳行后没有紧跟文本，则合并后续非时间戳行作为文本
        if not text:
            i += 1
            buffer = []
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    if buffer:
                        break
                    i += 1
                    continue
                if TIMESTAMP_TEXT_RE.match(next_line):
                    break
                buffer.append(next_line)
                i += 1
            text = " ".join(buffer)
        else:
            i += 1

        try:
            start_sec = parse_timestamp(start_text)
        except ValueError:
            continue

        text = text.strip()
        if not text:
            continue
        entries.append((start_sec, text))

    entries.sort(key=lambda x: x[0])
    return entries


def build_text_map(entries):
    """
    构建 {格式化时间戳: 文本} 映射表。
    同时生成精确键（含毫秒）和粗略键（不含毫秒），以兼容 strip 前后的文件名。
    """
    result = {}
    for start_sec, text in entries:
        # 精确键：HH-MM-SS.mmm（用于精确匹配）
        key_exact = format_segment_timestamp(start_sec)
        result[key_exact] = text
        # 粗略键：HH-MM-SS（用于 strip 后的文件名匹配）
        key_rough = format_segment_timestamp_no_ms(start_sec)
        if key_rough not in result:
            result[key_rough] = text
    return result


def build_examples_from_audio_dir(
    audio_dir: Path,
    text_map: dict,
    max_duration_sec: float = 60.0,
):
    """
    遍历音频目录下所有 .wav 文件，与文本映射表匹配，生成样本列表。

    匹配策略：
    1. 精确匹配：用 HH-MM-SS.mmm 格式键查找
    2. 粗略匹配：用 HH-MM-SS 格式键查找（适配 strip 后的文件名）
    3. 模糊匹配：在 1 秒容差范围内找最近的文本
    """
    examples = []
    for wav_path in sorted(audio_dir.rglob("*.wav")):
        filename = wav_path.name
        if not wav_path.exists() or wav_path.stat().st_size <= 0:
            continue

        # 从文件名解析起止时间
        parsed = parse_segment_filename(filename)
        if parsed is None:
            continue
        start_sec, end_sec = parsed
        duration_sec = end_sec - start_sec
        if duration_sec > max_duration_sec:
            continue

        # 验证音频可读且时长 > 0
        try:
            audio_duration = librosa.get_duration(path=wav_path)
        except Exception:
            audio_duration = 0.0
        if audio_duration <= 0:
            continue

        # 尝试匹配文本：先精确匹配，再粗略匹配，最后模糊匹配
        key_exact = format_segment_timestamp(start_sec)
        key_rough = format_segment_timestamp_no_ms(start_sec)

        text = text_map.get(key_exact)
        if text is None:
            text = text_map.get(key_rough)
        if text is None:
            # 模糊匹配：在 1 秒容差内找最近的文本
            nearest = None
            best_diff = 1.0
            for ts_key, ts_text in text_map.items():
                try:
                    diff = abs(parse_segment_timestamp(ts_key) - start_sec)
                except ValueError:
                    continue
                if diff < best_diff:
                    best_diff = diff
                    nearest = ts_text
            text = nearest

        if text is None:
            continue

        examples.append({
            "audio": wav_path.resolve().as_posix(),
            "text": text,
            "prompt": "",
        })
    return examples


def find_transcript_files(text_dir: Path, transcript_name: str = "transcript.txt"):
    """递归查找文本目录下所有指定名称的 transcript 文件。"""
    return sorted(text_dir.rglob(transcript_name))


def split_dataset(examples, train_ratio: float, seed: int = 42):
    """按比例随机划分训练集和验证集。"""
    random.Random(seed).shuffle(examples)
    split_at = int(len(examples) * train_ratio)
    return examples[:split_at], examples[split_at:]


def write_jsonl(examples, out_path: Path):
    """将样本列表写入 JSONL 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="根据 transcript.txt 和切分后的 wav 生成训练/验证 JSONL。"
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="切分后 wav 文件的根目录（递归搜索 *.wav）",
    )
    parser.add_argument(
        "--text-dir", required=True,
        help="transcript.txt 文件的根目录（递归搜索 transcript.txt）",
    )
    parser.add_argument(
        "--train-out", default="train.jsonl",
        help="输出训练集 JSONL 文件名，默认 train.jsonl",
    )
    parser.add_argument(
        "--val-out", default="val.jsonl",
        help="输出验证集 JSONL 文件名，默认 val.jsonl",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="输出 JSONL 文件的目录，默认当前目录",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9,
        help="训练集比例，默认 0.9",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--max-duration-sec", type=float, default=60.0,
        help="仅保留时长不超过该秒数的样本，默认 60",
    )
    args = parser.parse_args()

    # 验证输入目录
    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"文本目录不存在: {text_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 收集所有 transcript.txt 并解析文本条目
    transcripts = find_transcript_files(text_dir)
    if not transcripts:
        raise FileNotFoundError(f"在 {text_dir} 中未找到任何 transcript.txt 文件。")

    all_text_entries = []
    for transcript_path in transcripts:
        print(f"读取 transcript: {transcript_path}")
        entries = parse_transcript_file(transcript_path)
        all_text_entries.extend(entries)

    if not all_text_entries:
        raise ValueError("未解析到任何时间戳文本条目，请检查 transcript 文件格式。")

    # 2. 构建文本映射表并匹配音频
    text_map = build_text_map(all_text_entries)
    print(f"文本条目总数: {len(all_text_entries)}")

    examples = build_examples_from_audio_dir(audio_dir, text_map, args.max_duration_sec)
    if not examples:
        raise ValueError(
            "未生成任何训练样本，请检查音频文件名与 transcript 时间戳是否匹配。"
        )

    # 3. 划分训练集/验证集并输出
    train_examples, val_examples = split_dataset(examples, args.train_ratio, args.seed)

    write_jsonl(train_examples, output_dir / args.train_out)
    write_jsonl(val_examples, output_dir / args.val_out)

    print(
        f"生成完成: transcript 文件={len(transcripts)} "
        f"总样本={len(examples)} "
        f"train={len(train_examples)} val={len(val_examples)}"
    )


if __name__ == "__main__":
    main()
