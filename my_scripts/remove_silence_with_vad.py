#!/usr/bin/env python3
"""用 FireRedVAD 检测语音，再去掉静音片段并拼接成新的音频。

示例：
    python my_scripts/remove_silence_with_vad.py input.wav output.wav
    python my_scripts/remove_silence_with_vad.py ./audio_dir ./output_dir --exts .wav,.flac

逻辑：
1. 对音频做 VAD 检测，得到语音区间；
2. 对每个语音区间左右各扩展 0.5 秒作为停顿；
3. 把这些片段依次拼接起来，生成一个新的音频。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# 让脚本可以直接使用仓库里的 fireredvad 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fireredvad"))
from fireredvad.vad import FireRedVad, FireRedVadConfig


DEFAULT_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 VAD 去掉静音并拼接音频")
    parser.add_argument("input", help="输入音频文件或目录")
    parser.add_argument("output", help="输出音频文件或目录")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parents[1] / "fireredvad" / "VAD"),
        help="FireRedVAD 模型目录，默认使用仓库自带模型",
    )
    parser.add_argument("--pad", type=float, default=0.5, help="每段语音前后保留的停顿时长（秒），默认 0.5")
    parser.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)), help="当输入是目录时处理的音频后缀，逗号分隔")
    parser.add_argument("--speech-threshold", type=float, default=0.4, help="VAD 语音阈值，默认 0.4")
    parser.add_argument("--skip-existing", type=int, default=1,
                        help="增量模式: 跳过输出已存在的文件（默认 1），设 0 则强制重新处理")
    parser.add_argument("--fail-log", type=str, default="",
                        help="错误日志文件路径，记录处理失败的文件")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理所有文件（等效于 --skip-existing 0）")
    return parser.parse_args()


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(path, audio, sr)


def iter_audio_files(input_path: str, exts: set[str]) -> Iterable[Path]:
    path = Path(input_path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            yield file_path


def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    target_len = int(len(audio) * target_sr / sr)
    return resample_poly(audio, target_sr, sr)[:target_len].astype(np.float32)


def detect_speech_segments(audio_path: str, model_dir: str, pad: float, speech_threshold: float) -> List[Tuple[float, float]]:
    config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=speech_threshold,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000)
    vad = FireRedVad.from_pretrained(model_dir, config)
    result, _ = vad.detect(str(audio_path), do_postprocess=True)
    
    # 获取音频总时长
    _, sr = sf.read(str(audio_path))
    duration = result.get("dur", 0)
    
    # 先收集原始语音片段（不填充）
    raw_segments = list(result.get("timestamps", []))
    if not raw_segments:
        return []
    
    # 对每个片段应用 pad，但要检查相邻片段的距离
    segments = []
    for i, (start, end) in enumerate(raw_segments):
        # 获取下一个片段的起始位置（如果存在）
        next_start = raw_segments[i + 1][0] if i + 1 < len(raw_segments) else None
        
        # 检查当前片段的后部分是否应该填充
        padded_end = end + pad
        if next_start is not None and (next_start - end) <= 2 * pad:
            # 中间的静音不超过 2*pad，不填充
            padded_end = end
        else:
            padded_end = min(duration, padded_end)
        
        # 检查当前片段的前部分是否应该填充
        padded_start = start - pad
        if i > 0:
            prev_end = segments[-1][1]
            if (start - prev_end) <= 2 * pad:
                # 前面与上一个片段距离太近，不填充
                padded_start = start
            else:
                padded_start = max(0.0, padded_start)
        else:
            padded_start = max(0.0, padded_start)
        
        if padded_end > padded_start:
            segments.append((padded_start, padded_end))
    
    return segments


def stitch_audio(audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)

    chunks = []
    for start, end in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        if end_sample <= start_sample:
            continue
        chunk = audio[start_sample:end_sample]
        chunks.append(chunk)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def process_file(input_path: Path, output_path: Path, model_dir: str, pad: float, speech_threshold: float) -> None:
    """处理单个音频文件。成功返回 None，失败抛出异常。"""
    # 加载音频
    audio, sr = load_audio(str(input_path))
    if len(audio) == 0:
        raise ValueError(f"音频文件为空或无法读取: {input_path}")

    # 检测语音片段
    segments = detect_speech_segments(str(input_path), model_dir, pad, speech_threshold)

    # 拼接音频
    stitched = stitch_audio(audio, sr, segments)

    # 保存结果
    save_audio(str(output_path), stitched, sr)
    print(f"已处理: {input_path} -> {output_path}，检测到 {len(segments)} 段语音，"
          f"时长 {len(audio) / sr:.1f}s -> {len(stitched) / sr:.1f}s")


def main() -> int:
    args = parse_args()

    # --force 覆盖 --skip-existing
    skip_existing = not args.force and bool(args.skip_existing)

    input_path = Path(args.input)
    output_path = Path(args.output)

    exts = {e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e}

    # 单文件模式
    if input_path.is_file():
        if output_path.is_dir():
            output_path = output_path / input_path.name
        try:
            process_file(input_path, output_path, args.model_dir, args.pad, args.speech_threshold)
            return 0
        except Exception as e:
            print(f"[错误] {input_path}: {e}", file=sys.stderr)
            return 1

    # 目录模式
    if not input_path.is_dir():
        print("输入必须是音频文件或目录", file=sys.stderr)
        return 1

    if output_path.exists() and not output_path.is_dir():
        print("输出路径必须是目录（当输入是目录时）", file=sys.stderr)
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # 打开失败日志文件（如果指定）
    fail_log = None
    if args.fail_log:
        fail_log_path = Path(args.fail_log)
        fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        fail_log = fail_log_path.open("a", encoding="utf-8")

    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_files: list[str] = []

    for file_path in iter_audio_files(str(input_path), exts):
        rel = file_path.relative_to(input_path)
        out_file = output_path / rel

        # --- 增量跳过：输出已存在且非空 ---
        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            skip_count += 1
            continue

        # --- 处理 ---
        try:
            process_file(file_path, out_file, args.model_dir, args.pad, args.speech_threshold)
            success_count += 1
        except Exception as e:
            fail_count += 1
            msg = f"[错误] {file_path} -> {out_file}: {e}"
            print(msg, file=sys.stderr)
            failed_files.append(str(file_path))
            if fail_log is not None:
                fail_log.write(f"{file_path}\t{out_file}\t{e}\n")

    if fail_log is not None:
        fail_log.close()

    # --- 最终统计 ---
    total = success_count + skip_count + fail_count
    parts = [f"成功 {success_count} 个"]
    if skip_count:
        parts.append(f"跳过 {skip_count} 个（已存在）")
    if fail_count:
        parts.append(f"失败 {fail_count} 个")
        if failed_files:
            print("\n失败文件列表:", file=sys.stderr)
            for f in failed_files:
                print(f"  {f}", file=sys.stderr)
    print(f"\n处理完成，共 {total} 个文件，{'，'.join(parts)}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
