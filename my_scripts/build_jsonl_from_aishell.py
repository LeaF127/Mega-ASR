#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_content_line(line: str):
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if not parts:
        return None

    audio_id = parts[0]
    if not audio_id.endswith(".wav"):
        return None

    text_content = " ".join(parts[1:])
    cleaned_chars = []
    for char in text_content:
        if char.isspace():
            continue
        if char.isascii() and char.isalnum():
            continue
        cleaned_chars.append(char)

    cleaned_text = "".join(cleaned_chars)
    if not cleaned_text:
        return None
    return audio_id, cleaned_text


def collect_examples(split_dir: Path, dataset_root: Path):
    content_path = split_dir / "content.txt"
    wav_dir = split_dir / "wav"
    examples = []

    if not content_path.is_file():
        return examples
    if not wav_dir.is_dir():
        return examples

    for raw_line in content_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_content_line(raw_line)
        if parsed is None:
            continue

        audio_id, text = parsed
        wav_path = wav_dir / audio_id[:7] /audio_id
        if not wav_path.is_file():
            continue

        examples.append({
            "audio": wav_path,
            "text": text,
            "prompt": "",
        })

    return examples


def write_jsonl(examples, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="从 AISHELL 的 content.txt 中读取音频 id 和对应文本，生成 JSONL。"
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "aishell"),
        help="AISHELL 数据根目录，默认是仓库根目录下的 aishell",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出 JSONL 的目录；默认是 <root>/ft_megaasr",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AISHELL 根目录不存在: {root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "ft_megaasr"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "test"]
    for split_name in split_names:
        split_dir = root / split_name
        if not split_dir.is_dir():
            print(f"跳过: 未找到目录 {split_dir}")
            continue

        examples = collect_examples(split_dir, root)
        if not examples:
            print(f"跳过: {split_dir} 中没有解析到有效样本")
            continue

        out_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(examples, out_path)
        print(f"已生成 {out_path}，样本数={len(examples)}")

    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
