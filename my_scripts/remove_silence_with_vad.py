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


def detect_speech_segments(audio: np.ndarray, sr: int, model_dir: str, pad: float, speech_threshold: float) -> List[Tuple[float, float]]:
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
    result, _ = vad.detect(audio, do_postprocess=True)
    segments = []
    duration = len(audio) / sr
    for start, end in result.get("timestamps", []):
        start = max(0.0, start - pad)
        end = min(duration, end + pad)
        if end > start:
            segments.append((start, end))
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
    audio, sr = load_audio(str(input_path))
    if sr != 16000:
        print(f"警告: {input_path} 的采样率是 {sr} Hz，VAD 模型默认使用 16k，脚本会先重采样到 16k 进行检测。")
        vad_audio = resample_audio(audio, sr, 16000)
        vad_sr = 16000
    else:
        vad_audio = audio
        vad_sr = sr

    segments = detect_speech_segments(vad_audio, vad_sr, model_dir, pad, speech_threshold)
    stitched = stitch_audio(audio, sr, segments)
    save_audio(str(output_path), stitched, sr)
    print(f"已处理: {input_path} -> {output_path}，检测到 {len(segments)} 段语音")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    exts = {e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e}

    if input_path.is_file():
        if output_path.is_dir():
            output_path = output_path / input_path.name
        process_file(input_path, output_path, args.model_dir, args.pad, args.speech_threshold)
        return 0

    if not input_path.is_dir():
        print("输入必须是音频文件或目录", file=sys.stderr)
        return 1

    if output_path.exists() and not output_path.is_dir():
        print("输出路径必须是目录（当输入是目录时）", file=sys.stderr)
        return 1

    output_path.mkdir(parents=True, exist_ok=True)
    for file_path in iter_audio_files(str(input_path), exts):
        rel = file_path.relative_to(input_path)
        out_file = output_path / rel
        process_file(file_path, out_file, args.model_dir, args.pad, args.speech_threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
