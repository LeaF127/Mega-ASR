#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path

TIMESTAMP_TEXT_RE = re.compile(
    r"^\s*(?:Speaker\s+\d+\s*[:,]?\s*)?(\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)(?:\s+)(.*)$",
    re.IGNORECASE,
)
SEGMENT_FILENAME_RE = re.compile(
    r"^(\d{2}-\d{2}-\d{2}\.\d{3})_(\d{2}-\d{2}-\d{2}\.\d{3})\.wav$"
)


def parse_timestamp(timestamp_text: str) -> float:
    normalized = timestamp_text.replace("，", ",").replace(",", ".")
    parts = normalized.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp format: {timestamp_text}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_segment_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{ms:03d}"


def parse_segment_filename(filename: str):
    match = SEGMENT_FILENAME_RE.match(filename)
    if not match:
        return None
    start = match.group(1)
    end = match.group(2)
    return parse_segment_timestamp(start), parse_segment_timestamp(end)


def parse_segment_timestamp(timestamp_text: str) -> float:
    normalized = timestamp_text.replace("-", ":")
    return parse_timestamp(normalized)


def parse_transcript_file(transcript_path: Path):
    entries = []
    with transcript_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = TIMESTAMP_TEXT_RE.match(line)
            if not match:
                continue
            start_text, text = match.groups()
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
    result = {}
    for start_sec, text in entries:
        key = format_segment_timestamp(start_sec)
        result[key] = text
    return result


def build_examples_from_split_folder(split_folder: Path, text_map):
    examples = []
    for wav_path in sorted(split_folder.glob("*.wav")):
        filename = wav_path.name
        parsed = parse_segment_filename(filename)
        if parsed is None:
            continue
        start_sec, end_sec = parsed
        key = format_segment_timestamp(start_sec)
        text = text_map.get(key)
        if text is None:
            # 兼容精度差异，寻找最接近的起始时间
            nearest = None
            best_diff = 1.0
            for ts_key, ts_text in text_map.items():
                diff = abs(parse_segment_timestamp(ts_key) - start_sec)
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


def split_dataset(examples, train_ratio: float, seed: int = 42):
    random.Random(seed).shuffle(examples)
    split_at = int(len(examples) * train_ratio)
    return examples[:split_at], examples[split_at:]


def write_jsonl(examples, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 transcript.txt 和切分后的 wav 生成训练/验证 JSONL。"
    )
    parser.add_argument("--transcript", required=True, help="transcript.txt 文件路径")
    parser.add_argument("--split-dir", required=True, help="切分后 wav 所在目录")
    parser.add_argument("--train-out", default="train.jsonl", help="输出训练集 JSONL 文件名")
    parser.add_argument("--test-out", default="test.jsonl", help="输出验证集 JSONL 文件名")
    parser.add_argument("--output-dir", default=".", help="输出 JSONL 文件目录")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例，默认 0.9")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    if not transcript_path.is_file():
        raise FileNotFoundError(f"transcript 文件不存在: {transcript_path}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"切分后的 wav 目录不存在: {split_dir}")

    entries = parse_transcript_file(transcript_path)
    if not entries:
        raise ValueError(f"未解析到任何时间戳文本: {transcript_path}")

    text_map = build_text_map(entries)
    examples = build_examples_from_split_folder(split_dir, text_map)
    if not examples:
        raise ValueError(f"未生成任何训练样本，请检查 split_dir 路径和 wav 文件名格式: {split_dir}")

    train_examples, test_examples = split_dataset(examples, args.train_ratio, args.seed)

    write_jsonl(train_examples, output_dir / args.train_out)
    write_jsonl(test_examples, output_dir / args.test_out)
    print(f"生成完成: train={len(train_examples)} test={len(test_examples)} 总计={len(examples)}")


if __name__ == "__main__":
    main()
