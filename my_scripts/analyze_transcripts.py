#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import unicodedata


CHINESE_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
]

JAPANESE_CHARACTERS = [
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),  # Halfwidth Katakana
]

ENGLISH_RANGES = [
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
]

TEXT_FILENAMES = {"transcript.txt", "transript.txt"}
KEYWORDS_SECTION_RE = re.compile(r'^\s*Keywords\s*:\s*', re.IGNORECASE)
SPEAKER_LINE_RE = re.compile(r'^\s*Speaker\s+\d+', re.IGNORECASE)
TIMESTAMP_LINE_RE = re.compile(
    r'^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?|\d{1,2}h\s*\d{1,2}min|\d+(?:\.\d+)?s)\b',
    re.IGNORECASE,
)


def is_in_ranges(cp, ranges):
    return any(start <= cp <= end for start, end in ranges)


def line_has_letter(line):
    for char in line:
        cp = ord(char)
        if is_in_ranges(cp, CHINESE_RANGES) or is_in_ranges(cp, JAPANESE_CHARACTERS):
            return True
        if is_in_ranges(cp, ENGLISH_RANGES):
            return True
        if unicodedata.category(char).startswith("L"):
            return True
    return False


def is_metadata_line(line):
    stripped = line.strip()
    if not stripped:
        return True
    if KEYWORDS_SECTION_RE.match(line):
        return True
    if SPEAKER_LINE_RE.match(line):
        return True
    if TIMESTAMP_LINE_RE.match(line):
        return True
    if not line_has_letter(line) and any(char.isdigit() for char in line):
        return True
    return False


def filter_transcript_text(text):
    filtered_lines = []
    in_keyword_section = False

    for line in text.splitlines():
        if in_keyword_section:
            if line.strip() == "":
                in_keyword_section = False
            continue

        if KEYWORDS_SECTION_RE.match(line):
            in_keyword_section = True
            continue

        if is_metadata_line(line):
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def split_sentences(text):
    parts = re.split(r'(?<=[。！？!\?；;])|\n+', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def classify_sentence(sentence):
    counts = count_language_chars(sentence)
    total = counts["chinese"] + counts["english"] + counts["japanese"]
    if total == 0:
        return None, counts
    if counts["japanese"] > 0:
        return "japanese", counts
    if counts["chinese"] == 0 and counts["english"] > 0:
        return "english", counts
    if counts["english"] == 0 and counts["chinese"] > 0:
        return "chinese", counts
    if counts["chinese"] >= counts["english"]:
        return "chinese", counts
    return "english", counts


def count_language_chars(text):
    counts = {"chinese": 0, "japanese": 0, "english": 0, "other": 0}

    for char in text:
        if char.isspace():
            continue
        cp = ord(char)
        if is_in_ranges(cp, CHINESE_RANGES):
            counts["chinese"] += 1
        elif is_in_ranges(cp, JAPANESE_CHARACTERS):
            counts["japanese"] += 0 if char in "ー─" else 1
        elif is_in_ranges(cp, ENGLISH_RANGES):
            counts["english"] += 1
        else:
            cat = unicodedata.category(char)
            if cat.startswith("L"):
                counts["other"] += 1
    return counts


def analyze_text(text):
    filtered_text = filter_transcript_text(text)
    sentences = split_sentences(filtered_text)
    sentence_counts = {"chinese": 0, "japanese": 0, "english": 0, "other": 0}

    for sentence in sentences:
        lang, counts = classify_sentence(sentence)
        if lang is None:
            continue
        sentence_counts[lang] += 1

    total = sentence_counts["chinese"] + sentence_counts["english"] + sentence_counts["japanese"]
    if total == 0:
        return "filtered", sentence_counts

    japanese_ratio = sentence_counts["japanese"] / total
    english_ratio = sentence_counts["english"] / total
    chinese_ratio = sentence_counts["chinese"] / total

    if sentence_counts["chinese"] == 0 and sentence_counts["english"] > 0:
        return "filtered", sentence_counts
    if japanese_ratio > 0.5:
        return "filtered", sentence_counts
    if english_ratio > 0.8:
        return "filtered", sentence_counts

    return "keep", sentence_counts


def find_transcript_file(dir_path):
    for filename in TEXT_FILENAMES:
        candidate = os.path.join(dir_path, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def copy_folder(source_folder, dest_root):
    dest_folder = os.path.join(dest_root, os.path.basename(source_folder))
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    shutil.copytree(source_folder, dest_folder)
    return dest_folder


def reset_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="按句子语言占比分类：大部分日语或英文目录过滤，少量日语仍保留。")
    parser.add_argument("root", nargs="?", default=".", help="要分析的根目录，默认当前目录")
    parser.add_argument("--keep-dir", default="kept_folders", help="保存保留文件夹的输出目录")
    parser.add_argument("--filtered-dir", default="filtered_folders", help="保存过滤文件夹的输出目录")
    parser.add_argument("--dry-run", action="store_true", help="仅打印结果，不复制文件夹")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    keep_root = os.path.abspath(os.path.join(root, args.keep_dir))
    filtered_root = os.path.abspath(os.path.join(root, args.filtered_dir))

    if args.dry_run:
        os.makedirs(keep_root, exist_ok=True)
        os.makedirs(filtered_root, exist_ok=True)
    else:
        reset_output_dir(keep_root)
        reset_output_dir(filtered_root)

    print(f"分析根目录: {root}")
    print(f"保留输出目录: {keep_root}")
    print(f"过滤输出目录: {filtered_root}\n")

    processed = 0
    kept = 0
    filtered = 0

    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.startswith(keep_root) or dirpath.startswith(filtered_root):
            continue
        transcript_path = find_transcript_file(dirpath)
        if not transcript_path:
            continue

        processed += 1
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        result, counts = analyze_text(text)
        line = f"{dirpath} -> {result.upper()} | chinese={counts['chinese']} english={counts['english']} japanese={counts['japanese']} other={counts['other']}"
        print(line)

        if args.dry_run:
            continue

        if result == "keep":
            copy_folder(dirpath, keep_root)
            kept += 1
        else:
            copy_folder(dirpath, filtered_root)
            filtered += 1

    print(f"\n已处理文件夹: {processed}")
    print(f"保留: {kept}")
    print(f"过滤: {filtered}")

if __name__ == "__main__":
    main()
