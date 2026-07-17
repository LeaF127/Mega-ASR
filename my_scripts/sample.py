#!/usr/bin/env python3
"""
从文本目录和音频目录中，全局随机抽样 N 条长度适中的音频及对应文本作为测试集。

目录结构要求:
    text_root/                 音频_root/
    ├── dir1/                  ├── dir1/
    │   └── text               │   └── *.wav / *.mp3 ...
    ├── dir2/                  ├── dir2/
    │   └── text               │   └── *.wav / *.mp3 ...
    └── ...                    └── ...

两个根目录下的一级子目录必须一致，脚本会自动校验并只处理共同存在的子目录。

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
    └── manifest.jsonl

用法:
    python my_scripts/sample_one_per_dir.py \
        --text-root /path/to/text_root \
        --audio-root /path/to/audio_root \
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

DEFAULT_TEXT_FILENAME = "text"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


# ---------------------------------------------------------------------------
# 文本解析
# ---------------------------------------------------------------------------

def parse_text_line(line: str):
    """解析 text 文件中的一行，返回 (audio_id, text_content) 或 None。"""
    line = line.strip()
    if not line:
        return None
    parts = line.split(None, 1)  # 只按第一个空白字符拆分
    if len(parts) < 2:
        return None
    audio_id = parts[0].strip()
    text_content = parts[1].strip()
    if not text_content:
        return None
    return audio_id, text_content


def load_text_map(text_dir: Path, text_filename: str = DEFAULT_TEXT_FILENAME) -> dict[str, str]:
    """
    读取子目录中的 text 文件，返回 {audio_id: text_content} 映射。
    audio_id 统一转为小写以便跨目录匹配。
    """
    text_path = text_dir / text_filename
    if not text_path.is_file():
        return {}

    text_map: dict[str, str] = {}
    for raw_line in text_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_text_line(raw_line)
        if parsed:
            audio_id, text_content = parsed
            text_map[audio_id.lower()] = text_content
    return text_map


# ---------------------------------------------------------------------------
# 音频查找 & 时长
# ---------------------------------------------------------------------------

def find_audio_file(dir_path: Path, audio_id: str):
    """在目录中查找匹配 audio_id 的音频文件（大小写不敏感）。"""
    audio_id_lower = audio_id.lower()

    # 精确匹配：audio_id + 扩展名
    for ext in AUDIO_EXTENSIONS:
        candidate = dir_path / f"{audio_id}{ext}"
        if candidate.is_file():
            return candidate

    # 如果 audio_id 已含扩展名
    direct = dir_path / audio_id
    if direct.is_file():
        return direct

    # 大小写不敏感兜底扫描（仅当前目录，不递归）
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            if f.stem.lower() == audio_id_lower:
                return f

    return None


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
# 主逻辑
# ---------------------------------------------------------------------------

def collect_candidates(
    text_root: Path,
    audio_root: Path,
    text_filename: str,
    min_duration: float,
    max_duration: float,
):
    """
    遍历所有共同子目录，收集符合条件的 (audio_path, text) 候选。

    返回: list[dict(audio, text, duration, subdir)]
    """
    text_subdirs = {d.name: d for d in text_root.iterdir() if d.is_dir()}
    audio_subdirs = {d.name: d for d in audio_root.iterdir() if d.is_dir()}

    common = sorted(set(text_subdirs) & set(audio_subdirs))
    if not common:
        print("❌ 文本目录和音频目录没有共同的一级子目录！")
        return []

    # 报告不对齐的目录
    only_text = set(text_subdirs) - set(audio_subdirs)
    only_audio = set(audio_subdirs) - set(text_subdirs)
    if only_text:
        print(f"⚠ 仅在文本目录存在: {sorted(only_text)}")
    if only_audio:
        print(f"⚠ 仅在音频目录存在: {sorted(only_audio)}")

    print(f"共同子目录数: {len(common)}")
    print()

    candidates = []
    total_audio_files = 0
    matched = 0
    skipped_no_audio = 0
    skipped_duration = 0

    for dir_name in common:
        text_dir = text_subdirs[dir_name]
        audio_dir = audio_subdirs[dir_name]

        text_map = load_text_map(text_dir, text_filename)
        if not text_map:
            continue

        # 遍历音频文件
        audio_files = [
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        total_audio_files += len(audio_files)

        for audio_path in audio_files:
            # 匹配文本：按 stem 查找
            audio_id_lower = audio_path.stem.lower()
            text_content = text_map.get(audio_id_lower)

            if text_content is None:
                # 尝试完整文件名匹配
                text_content = text_map.get(audio_path.name.lower())

            if text_content is None:
                skipped_no_audio += 1
                continue

            # 检查时长
            if min_duration > 0 or max_duration > 0:
                duration = get_audio_duration(audio_path)
                if duration is None:
                    if HAS_SOUNDFILE:
                        skipped_duration += 1
                        continue
                    duration = -1
                elif duration < min_duration or duration > max_duration:
                    skipped_duration += 1
                    continue
            else:
                duration = -1

            candidates.append({
                "audio": audio_path,
                "text": text_content,
                "duration": duration,
                "subdir": dir_name,
            })
            matched += 1

    print(f"音频文件总数: {total_audio_files}")
    print(f"成功匹配:     {matched}")
    print(f"无对应文本:   {skipped_no_audio}")
    print(f"时长不符:     {skipped_duration}  (范围: {min_duration}s – {max_duration}s)")
    print()

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="从文本/音频分离的目录结构中，全局随机抽样 N 条长度适中的音频+文本作为测试集。"
    )
    parser.add_argument("--text-root", required=True, help="文本根目录（含一级子目录，每个下有 text 文件）")
    parser.add_argument("--audio-root", required=True, help="音频根目录（含一级子目录，每个下有音频文件）")
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--num", type=int, default=20, help="抽样数量，默认 20")
    parser.add_argument("--min-duration", type=float, default=3.0, help="最小时长（秒），默认 3.0")
    parser.add_argument("--max-duration", type=float, default=30.0, help="最大时长（秒），默认 30.0")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--text-filename",
        default=DEFAULT_TEXT_FILENAME,
        help=f"文本文件名，默认: {DEFAULT_TEXT_FILENAME}",
    )

    args = parser.parse_args()

    if not HAS_SOUNDFILE:
        print("⚠ 未安装 soundfile，将跳过时长过滤。安装: pip install soundfile")

    text_root = Path(args.text_root).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not text_root.is_dir():
        raise SystemExit(f"文本根目录不存在: {text_root}")
    if not audio_root.is_dir():
        raise SystemExit(f"音频根目录不存在: {audio_root}")

    if args.seed is not None:
        random.seed(args.seed)

    print(f"文本根目录: {text_root}")
    print(f"音频根目录: {audio_root}")
    print(f"输出目录:   {output_dir}")
    print(f"目标数量:   {args.num}")
    print(f"时长范围:   {args.min_duration}s – {args.max_duration}s")
    if args.seed is not None:
        print(f"随机种子:   {args.seed}")
    print()

    # 收集候选
    candidates = collect_candidates(
        text_root=text_root,
        audio_root=audio_root,
        text_filename=args.text_filename,
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

    # 按子目录+文件名排序，保证输出顺序稳定
    sampled.sort(key=lambda x: (x["subdir"], x["audio"].name))

    # 创建输出子目录
    audio_out = output_dir / "audio"
    text_out = output_dir / "text"
    audio_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    for idx, item in enumerate(sampled, start=1):
        seq_id = f"{idx:03d}"
        src_audio = item["audio"]

        # 复制音频，统一扩展名
        dst_audio = audio_out / f"{seq_id}{src_audio.suffix}"
        shutil.copy2(src_audio, dst_audio)

        # 写入文本
        dst_text = text_out / f"{seq_id}.txt"
        dst_text.write_text(item["text"], encoding="utf-8")

        # 记录清单
        manifest_records.append({
            "audio": str(dst_audio.resolve()),
            "answer": item["text"],
            "duration": round(item["duration"], 2) if item["duration"] > 0 else None,
            "source_subdir": item["subdir"],
            "source_audio": str(src_audio),
        })

        dur_str = f"{item['duration']:.1f}s" if item["duration"] > 0 else "?"
        print(f"  [{seq_id}] {item['subdir']}/{src_audio.name}  ({dur_str})")

    # 写入汇总清单
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计时长
    valid_durations = [r["duration"] for r in manifest_records if r["duration"] is not None and r["duration"] > 0]
    total_dur = sum(valid_durations)

    print()
    print("=" * 55)
    print(f"✅ 完成！")
    print(f"   抽取数量: {len(sampled)} 条")
    if valid_durations:
        print(f"   总时长:   {total_dur:.1f}s  ({total_dur/60:.1f} min)")
        print(f"   平均时长: {total_dur/len(valid_durations):.1f}s")
    print(f"   音频目录: {audio_out}")
    print(f"   文本目录: {text_out}")
    print(f"   汇总清单: {manifest_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
