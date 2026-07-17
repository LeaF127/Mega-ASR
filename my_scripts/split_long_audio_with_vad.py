#!/usr/bin/env python3
"""
对超长音频（默认 >60s）进行 VAD 导语切分，并将对应文本按比例切分，
输出新的 JSONL 数据文件。

核心策略：
1. 读取 JSONL，支持训练格式 (audio/text/prompt) 和评估格式 (audio/answer)
2. 对每个超长音频运行 FireRedVAD，获取语音/静默区间
3. 贪心分组语音段，使每组总时长 ≤ 60s，在静默最深处断开
4. 按音频时长比例切分对应文本，并尝试在句子边界处切开
5. 输出切分后的音频片 + 新 JSONL

用法:
    python my_scripts/split_long_audio_with_vad.py \
        --input_jsonl data/long_audios.jsonl \
        --output_jsonl data/split_output.jsonl \
        --audio_output_dir data/audio_chunks/ \
        --min_duration 60

依赖: soundfile, numpy, tqdm, scipy
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ─── FireRedVAD ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fireredvad"))
from fireredvad.vad import FireRedVad, FireRedVadConfig  # noqa: E402

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_SEGMENT_DURATION = 60.0       # 每个输出片段最长时间（秒）
ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "
SENTENCE_BOUNDARY_RE = re.compile(r"[。！？；.!?;\n]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 导语切分超长音频并自动切分对应文本"
    )
    parser.add_argument(
        "--input_jsonl", required=True,
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_jsonl", required=True,
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--audio_output_dir", required=True,
        help="切分后音频片段输出目录",
    )
    parser.add_argument(
        "--min_duration", type=float, default=60.0,
        help="最小时长阈值（秒），超过此值的音频才会被切分，默认 60",
    )
    parser.add_argument(
        "--vad_model_dir", type=str, default=None,
        help="FireRedVAD 模型目录（默认使用仓库自带的 VAD 模型）",
    )
    parser.add_argument(
        "--speech_threshold", type=float, default=0.4,
        help="VAD 语音检测阈值，默认 0.4",
    )
    parser.add_argument(
        "--max_segment_duration", type=float, default=MAX_SEGMENT_DURATION,
        help="切分后每段的最长时间（秒），默认 60",
    )
    parser.add_argument(
        "--pad", type=float, default=0.3,
        help="切分点前后保留的停顿时长（秒），默认 0.3",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="跳过已存在的输出音频文件（默认开启）",
    )
    parser.add_argument(
        "--no_skip_existing", action="store_false", dest="skip_existing",
        help="强制重新处理所有文件",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="仅预览切分计划，不实际切分",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  VAD 检测
# ═══════════════════════════════════════════════════════════════════════════════

def get_vad_segments(
    audio_path: str,
    model_dir: str,
    speech_threshold: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """运行 FireRedVAD，返回语音片段列表 [(start, end), ...] 和音频总时长（秒）。"""
    config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=speech_threshold,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    vad = FireRedVad.from_pretrained(model_dir, config)
    result, _ = vad.detect(str(audio_path), do_postprocess=True)
    timestamps: List[Tuple[float, float]] = list(result.get("timestamps", []))
    duration: float = result.get("dur", 0.0)
    return timestamps, duration


# ═══════════════════════════════════════════════════════════════════════════════
#  语音段贪心分组
# ═══════════════════════════════════════════════════════════════════════════════

def group_segments_into_chunks(
    segments: List[Tuple[float, float]],
    max_duration: float,
) -> List[List[Tuple[float, float]]]:
    """将 VAD 语音段贪心分组，使每组总时长 ≤ max_duration。"""
    if not segments:
        return []

    chunks: List[List[Tuple[float, float]]] = []
    current_chunk = [segments[0]]
    current_dur = segments[0][1] - segments[0][0]

    for seg in segments[1:]:
        seg_dur = seg[1] - seg[0]
        gap = seg[0] - current_chunk[-1][1]

        if current_dur + gap + seg_dur > max_duration:
            # 当前片段本身就很长，直接开新组
            chunks.append(current_chunk)
            current_chunk = [seg]
            current_dur = seg_dur
        else:
            current_chunk.append(seg)
            current_dur += gap + seg_dur

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def find_best_split_point(
    prev_end: float,
    next_start: float,
) -> float:
    """在两个语音段之间的静默间隙中找到最佳切分点（中点）。"""
    return (prev_end + next_start) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
#  音频切分
# ═══════════════════════════════════════════════════════════════════════════════

def split_audio_at_times(
    audio: np.ndarray,
    sr: int,
    split_times: List[float],
) -> List[np.ndarray]:
    """根据切分时间点列表切分音频。"""
    if not split_times:
        return [audio]

    chunks: List[np.ndarray] = []
    prev_sample = 0
    for t in split_times:
        split_sample = int(round(t * sr))
        # 确保边界有效
        split_sample = max(prev_sample + 1, min(split_sample, len(audio)))
        chunk = audio[prev_sample:split_sample]
        if len(chunk) > 0:
            chunks.append(chunk)
        prev_sample = split_sample

    # 最后一段
    last_chunk = audio[prev_sample:]
    if len(last_chunk) > 0:
        chunks.append(last_chunk)

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  文本切分
# ═══════════════════════════════════════════════════════════════════════════════

def extract_transcript(text: str) -> Tuple[str, str]:
    """从训练格式文本中提取纯转录内容。
    
    返回 (prefix, transcript)，其中 prefix 是 'language XX<asr_text>' 部分。
    如果不存在该标记，prefix 为空字符串，transcript 为原文本。
    """
    if ASR_TEXT_TAG in text:
        idx = text.index(ASR_TEXT_TAG) + len(ASR_TEXT_TAG)
        prefix = text[:idx]
        transcript = text[idx:]
        return prefix, transcript
    return "", text


def find_nearest_sentence_boundary(
    text: str,
    target_char_idx: int,
    search_window: int = 30,
) -> Optional[int]:
    """在 target_char_idx 附近寻找最近的句子边界。
    
    优先向前找，找不到再向后找。
    """
    # 向前搜索
    start = max(0, target_char_idx - search_window)
    before = text[start:target_char_idx]
    for m in SENTENCE_BOUNDARY_RE.finditer(before):
        pass  # 迭代到最后一次匹配
    # 从后往前找最近的边界
    for i in range(target_char_idx - 1, start - 1, -1):
        if SENTENCE_BOUNDARY_RE.match(text[i]):
            return i + 1  # 在边界后切开

    # 向后搜索
    end = min(len(text), target_char_idx + search_window)
    for i in range(target_char_idx, end):
        if SENTENCE_BOUNDARY_RE.match(text[i]):
            return i + 1

    return None


def split_text_by_ratio(
    transcript: str,
    ratios: List[float],
) -> List[str]:
    """按比例切分文本，尽量在句子边界处切开。"""
    if len(ratios) <= 1:
        return [transcript]

    total_len = len(transcript)
    chunks: List[str] = []
    prev_idx = 0

    # 累积比例得到每个切分点的目标字符位置
    cum_ratio = 0.0
    for ratio in ratios[:-1]:  # 最后一段不需要目标位置
        cum_ratio += ratio
        target_idx = int(round(cum_ratio * total_len))
        target_idx = max(prev_idx + 1, min(target_idx, total_len))

        # 尝试在句子边界处切开
        boundary = find_nearest_sentence_boundary(transcript, target_idx)
        if boundary is not None and prev_idx < boundary < total_len:
            split_idx = boundary
        else:
            split_idx = target_idx

        chunk = transcript[prev_idx:split_idx]
        if chunk:
            chunks.append(chunk)
        prev_idx = split_idx

    # 最后一段
    last_chunk = transcript[prev_idx:]
    if last_chunk:
        chunks.append(last_chunk)

    return chunks


def split_full_text(
    full_text: str,
    ratios: List[float],
) -> List[str]:
    """对完整文本进行切分，自动处理训练格式中的 prefix。"""
    prefix, transcript = extract_transcript(full_text)
    if not transcript.strip():
        return [full_text]

    transcript_chunks = split_text_by_ratio(transcript, ratios)

    if prefix:
        return [prefix + chunk for chunk in transcript_chunks]
    return transcript_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  核心处理逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def get_resolved_vad_model_dir(args: argparse.Namespace) -> str:
    """确定 VAD 模型路径。"""
    if args.vad_model_dir is not None:
        return args.vad_model_dir
    # 默认使用仓库自带的 VAD 模型
    default = str(Path(__file__).resolve().parents[1] / "fireredvad" / "VAD")
    if not os.path.isdir(default):
        raise FileNotFoundError(
            f"默认 VAD 模型目录不存在: {default}\n"
            f"请使用 --vad_model_dir 指定正确的模型路径"
        )
    return default


def get_audio_duration_ffmpeg(audio_path: str) -> Optional[float]:
    """使用 ffprobe 快速获取音频时长，作为前置过滤（避免加载整个音频）。"""
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def get_audio_duration_soundfile(audio_path: str) -> Optional[float]:
    """使用 soundfile 获取音频时长。"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return None


def process_one_entry(
    entry: Dict[str, Any],
    entry_idx: int,
    args: argparse.Namespace,
    vad_model_dir: str,
) -> List[Dict[str, Any]]:
    """处理一条 JSONL 条目，返回切分后的条目列表。"""
    audio_path = entry.get("audio") or entry.get("audio_path")
    if not audio_path:
        tqdm.write(f"[{entry_idx}] 跳过：没有音频路径")
        return [entry]

    if not os.path.isfile(audio_path):
        tqdm.write(f"[{entry_idx}] 跳过：音频文件不存在: {audio_path}")
        return [entry]

    # ── 获取时长 ──────────────────────────────────────────────────────────
    duration = get_audio_duration_soundfile(audio_path)
    if duration is None:
        duration = get_audio_duration_ffmpeg(audio_path)
    if duration is None:
        tqdm.write(f"[{entry_idx}] 跳过：无法获取音频时长: {audio_path}")
        return [entry]

    # 时长未超过阈值，不切分
    if duration <= args.min_duration:
        return [entry]

    # ── 获取文本 ──────────────────────────────────────────────────────────
    text_field = "text" if "text" in entry else "answer"
    full_text = entry.get(text_field, "")
    if not full_text.strip():
        tqdm.write(f"[{entry_idx}] 跳过 {os.path.basename(audio_path)}：无文本")
        return [entry]

    # ── VAD 检测 ─────────────────────────────────────────────────────────
    segments, detected_dur = get_vad_segments(audio_path, vad_model_dir, args.speech_threshold)
    if not segments:
        tqdm.write(f"[{entry_idx}] 跳过 {os.path.basename(audio_path)}：VAD 未检测到语音")
        return [entry]

    # ── 贪心分组 ─────────────────────────────────────────────────────────
    chunks = group_segments_into_chunks(segments, args.max_segment_duration)

    if len(chunks) <= 1:
        # VAD 分组后也只有一段，说明自然停顿不足以切分
        # 如果时长仍超过限制，强制在比例位置切开
        if duration <= args.max_segment_duration:
            return [entry]
        # 按目标时长强制切分
        num_splits = int(np.ceil(duration / args.max_segment_duration))
        chunks = []
        for i in range(num_splits):
            start_time = i * args.max_segment_duration
            end_time = min((i + 1) * args.max_segment_duration, duration)
            # 在 VAD 片段中找对应的语音段
            chunk_segs = [s for s in segments if s[0] < end_time and s[1] > start_time]
            if chunk_segs:
                chunks.append(chunk_segs)

        if not chunks:
            return [entry]

    # ── 计算切分点 ───────────────────────────────────────────────────────
    split_times: List[float] = []
    for i in range(len(chunks) - 1):
        prev_end = chunks[i][-1][1]
        next_start = chunks[i + 1][0][0]
        split_time = find_best_split_point(prev_end, next_start)
        split_times.append(split_time)

    # ── 计算各段时长 ─────────────────────────────────────────────────────
    chunk_durations: List[float] = []
    for i, chunk_segs in enumerate(chunks):
        start_t = chunk_segs[0][0]
        end_t = chunk_segs[-1][1]
        chunk_durations.append(end_t - start_t)

    # ── 切分文本 ─────────────────────────────────────────────────────────
    total_chunk_dur = sum(chunk_durations)
    ratios = [d / total_chunk_dur for d in chunk_durations]
    text_chunks = split_full_text(full_text, ratios)

    # 如果文本切分段数与音频段数不匹配（极端情况），补齐
    while len(text_chunks) < len(chunks):
        text_chunks.append(text_chunks[-1] if text_chunks else full_text)
    while len(text_chunks) > len(chunks):
        text_chunks[-2] = text_chunks[-2] + text_chunks[-1]
        text_chunks.pop()

    # ── dry-run 模式 ─────────────────────────────────────────────────────
    if args.dry_run:
        for i, (chunk_segs, t_chunk) in enumerate(zip(chunks, text_chunks)):
            chunk_dur = chunk_durations[i]
            _, transcript = extract_transcript(t_chunk)
            tqdm.write(
                f"  [{i + 1}] {chunk_dur:.1f}s | "
                f"文本 {len(transcript)} 字 | "
                f"{transcript[:50]}{'...' if len(transcript) > 50 else ''}"
            )
        return []

    # ── 加载音频并切分 ───────────────────────────────────────────────────
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    audio_chunks = split_audio_at_times(audio, sr, split_times)

    # ── 生成输出条目 ─────────────────────────────────────────────────────
    audio_basename = Path(audio_path).stem
    audio_output_dir = Path(args.audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    new_entries: List[Dict[str, Any]] = []
    for i, (audio_chunk, text_chunk) in enumerate(zip(audio_chunks, text_chunks)):
        out_filename = f"{audio_basename}_part{i + 1:03d}.wav"
        out_path = audio_output_dir / out_filename

        # 保存音频
        sf.write(str(out_path), audio_chunk, sr)

        # 构建新条目
        new_entry = dict(entry)
        new_entry["audio"] = str(out_path.resolve())
        new_entry[text_field] = text_chunk
        if "duration" in new_entry:
            new_entry["duration"] = round(len(audio_chunk) / sr, 3)
        new_entries.append(new_entry)

    return new_entries


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()

    # ── 确认 VAD 模型路径 ─────────────────────────────────────────────────
    try:
        vad_model_dir = get_resolved_vad_model_dir(args)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    # ── 读取输入 JSONL ────────────────────────────────────────────────────
    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        print(f"错误: 输入文件不存在: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"读取 {len(entries)} 条记录")

    # ── 逐个处理 ──────────────────────────────────────────────────────────
    output_entries: List[Dict[str, Any]] = []
    total_before = 0
    total_after = 0
    skipped_no_vad = 0

    pbar = tqdm(entries, desc="处理中", unit="条", ncols=80)
    for idx, entry in enumerate(pbar):
        total_before += 1
        result = process_one_entry(entry, idx, args, vad_model_dir)
        if not result:
            # dry-run 模式：结果为空列表
            continue
        output_entries.extend(result)
        total_after += len(result)
        if len(result) > 1:
            pbar.set_postfix_str(f"切分 {len(result)} 段")

    # ── 写入输出 JSONL ────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n预览完成。共 {total_before} 条中需要切分的已展示如上。")
        return 0

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── 统计 ──────────────────────────────────────────────────────────────
    split_count = sum(1 for e in output_entries if "_part" in e.get("audio", ""))
    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"  原始条数:      {total_before}")
    print(f"  输出条数:      {total_after}")
    print(f"  其中切分条数:  {split_count}")
    print(f"  新增条数:      {total_after - total_before}")
    print(f"  输出 JSONL:    {output_path.resolve()}")
    print(f"  音频目录:      {Path(args.audio_output_dir).resolve()}")
    print(f"{'=' * 50}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
