#!/usr/bin/env python3
"""
从已生成的 train.jsonl / val.jsonl 中随机抽样 N 条长度适中的音频及对应文本作为测试集。

JSONL 每行格式:
    {"audio": "/path/to/audio.wav", "text": " transcript text", "prompt": ""}

输出结构:
    output/
    ├── audio/
    │   ├── 001.wav
    │   ├── 002.wav
    │   └── ...
    ├── text/
    │   ├── 001.txt
    │   ├── 002.txt
    │   └── ...
    └── test.jsonl          # 评估格式的 JSONL: {"audio": "...", "answer": "..."}

用法:
    # 从单个 JSONL 抽样
    python my_scripts/sample.py \
        --jsonl train.jsonl \
        --output /path/to/testset \
        --num 20

    # 合并多个 JSONL 后抽样
    python my_scripts/sample.py \
        --jsonl train.jsonl val.jsonl \
        --output /path/to/testset \
        --num 20 \
        --min-duration 3.0 \
        --max-duration 30.0 \
        --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# ---------------------------------------------------------------------------
# 音频时长
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: Path) -> float | None:
    """获取音频时长（秒），失败返回 None。"""
    if not HAS_SOUNDFILE:
        return None
    try:
        info = sf.info(str(audio_path))
        return float(info.duration)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSONL 读取 & 过滤
# ---------------------------------------------------------------------------

def load_jsonl(jsonl_path: Path):
    """读取 JSONL 文件，返回 list[dict]。"""
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  ⚠ 跳过 {jsonl_path.name} 第 {line_no} 行: JSON 解析失败 ({exc})")
                continue
            records.append(rec)
    return records


def collect_candidates(
    jsonl_paths: list[Path],
    min_duration: float,
    max_duration: float,
):
    """
    从多个 JSONL 中读取所有条目，按音频时长过滤。

    返回: list[dict(audio, text, duration, source_jsonl)]
    """
    candidates = []
    total = 0
    skipped_missing_audio = 0
    skipped_duration = 0
    skipped_no_text = 0

    for jsonl_path in jsonl_paths:
        records = load_jsonl(jsonl_path)
        print(f"读取 {jsonl_path.name}: {len(records)} 条")

        for rec in records:
            total += 1
            audio_path_str = rec.get("audio", "")
            text = rec.get("text", "").strip()

            if not audio_path_str:
                skipped_missing_audio += 1
                continue
            if not text:
                skipped_no_text += 1
                continue

            audio_path = Path(audio_path_str)
            if not audio_path.is_file():
                skipped_missing_audio += 1
                continue

            # 检查时长
            duration = None
            if min_duration > 0 or max_duration > 0:
                duration = get_audio_duration(audio_path)
                if duration is None:
                    skipped_duration += 1
                    continue
                if duration < min_duration or duration > max_duration:
                    skipped_duration += 1
                    continue

            candidates.append({
                "audio": audio_path,
                "text": text,
                "duration": duration,
                "source_jsonl": jsonl_path.name,
            })

    print()
    print(f"JSONL 总条目:   {total}")
    print(f"音频缺失/不可读: {skipped_missing_audio}")
    print(f"文本为空:       {skipped_no_text}")
    print(f"时长不符:       {skipped_duration}  (范围: {min_duration}s – {max_duration}s)")
    print(f"候选数量:       {len(candidates)}")
    print()

    return candidates


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="从 train.jsonl / val.jsonl 中随机抽样 N 条长度适中的音频+文本作为测试集。"
    )
    parser.add_argument(
        "--jsonl", nargs="+", required=True,
        help="输入 JSONL 文件，可指定多个（如 train.jsonl val.jsonl）",
    )
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--num", type=int, default=20, help="抽样数量，默认 20")
    parser.add_argument("--min-duration", type=float, default=3.0, help="最小时长（秒），默认 3.0")
    parser.add_argument("--max-duration", type=float, default=30.0, help="最大时长（秒），默认 30.0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")

    args = parser.parse_args()

    if not HAS_SOUNDFILE:
        print("⚠ 未安装 soundfile，将跳过时长过滤。安装: pip install soundfile")

    jsonl_paths = [Path(p).expanduser().resolve() for p in args.jsonl]
    for p in jsonl_paths:
        if not p.is_file():
            raise SystemExit(f"JSONL 文件不存在: {p}")

    output_dir = Path(args.output).expanduser().resolve()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"输入 JSONL:  {[p.name for p in jsonl_paths]}")
    print(f"输出目录:    {output_dir}")
    print(f"目标数量:    {args.num}")
    print(f"时长范围:    {args.min_duration}s – {args.max_duration}s")
    print(f"随机种子:    {args.seed}")
    print()

    # 收集候选
    candidates = collect_candidates(
        jsonl_paths=jsonl_paths,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    if not candidates:
        raise SystemExit("没有找到任何符合条件的音频-文本对。")

    # 随机抽样
    if len(candidates) < args.num:
        print(f"⚠ 候选数量 ({len(candidates)}) 不足目标数量 ({args.num})，将全部抽取。")
    sample_n = min(args.num, len(candidates))
    sampled = random.sample(candidates, sample_n)

    # 按来源文件名+音频名排序
    sampled.sort(key=lambda x: (x["source_jsonl"], x["audio"].name))

    # 创建输出子目录
    audio_out = output_dir / "audio"
    text_out = output_dir / "text"
    audio_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)

    test_records = []

    for idx, item in enumerate(sampled, start=1):
        seq_id = f"{idx:03d}"
        src_audio = item["audio"]

        # 复制音频
        dst_audio = audio_out / f"{seq_id}{src_audio.suffix}"
        shutil.copy2(src_audio, dst_audio)

        # 写入文本
        dst_text = text_out / f"{seq_id}.txt"
        dst_text.write_text(item["text"], encoding="utf-8")

        # 评估格式记录
        test_records.append({
            "audio": str(dst_audio.resolve()),
            "answer": item["text"],
            "duration": round(item["duration"], 2) if item["duration"] is not None else None,
            "source_jsonl": item["source_jsonl"],
            "source_audio": str(src_audio),
        })

        dur_str = f"{item['duration']:.1f}s" if item["duration"] is not None else "?"
        print(f"  [{seq_id}] {src_audio.name}  ({dur_str})  ← {item['source_jsonl']}")

    # 写入 test.jsonl（评估格式）
    test_jsonl_path = output_dir / "test.jsonl"
    with test_jsonl_path.open("w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计时长
    valid_durations = [r["duration"] for r in test_records if r["duration"] is not None and r["duration"] > 0]
    total_dur = sum(valid_durations)

    print()
    print("=" * 55)
    print("✅ 完成！")
    print(f"   抽取数量: {len(sampled)} 条")
    if valid_durations:
        print(f"   总时长:   {total_dur:.1f}s  ({total_dur / 60:.1f} min)")
        print(f"   平均时长: {total_dur / len(valid_durations):.1f}s")
    print(f"   音频目录: {audio_out}")
    print(f"   文本目录: {text_out}")
    print(f"   测试清单: {test_jsonl_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
