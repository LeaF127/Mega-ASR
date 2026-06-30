#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

def setup_paths():
    root = Path(__file__).resolve().parent.parent
    sft_path = root / "src" / "MegaASR" / "A2S-SFT"
    sys.path.insert(0, str(sft_path))


def make_sine_wave(duration_s: float, sr: int = 16000, freq: float = 440.0):
    samples = int(round(duration_s * sr))
    t = np.linspace(0.0, duration_s, num=samples, endpoint=False, dtype=np.float32)
    return 0.1 * np.sin(2.0 * np.pi * freq * t, dtype=np.float32)


def save_audio(path: Path, waveform: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr, subtype="PCM_16")


def parse_args():
    p = argparse.ArgumentParser(description="Test fine-tuning audio duration limits for Mega-ASR A2S-SFT")
    p.add_argument("--model_path", type=str, required=True, help="Qwen3-ASR base model path")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate for fine-tuning audio")
    p.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1200.0],
        help="Audio durations in seconds to test",
    )
    p.add_argument("--text", type=str, default="测试语音长度上限", help="Dummy transcript text")
    p.add_argument("--tmp_dir", type=str, default="./tmp_finetune_audio_test", help="Temporary directory for generated audio")
    return p.parse_args()


def main():
    setup_paths()
    from dataloader import Qwen3ASRCollator, read_audio
    from modeling import load_qwen3_asr

    args = parse_args()
    args.tmp_dir = Path(args.tmp_dir).resolve()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and processor from", args.model_path)
    model, processor, _ = load_qwen3_asr(args.model_path)
    collator = Qwen3ASRCollator(processor=processor, sampling_rate=args.sr)

    results = []
    for duration in args.durations:
        audio_path = args.tmp_dir / f"audio_{int(duration)}s.wav"
        print(f"\n=== Testing duration {duration:.0f}s ===")
        waveform = make_sine_wave(duration, sr=args.sr)
        save_audio(audio_path, waveform, args.sr)
        print(f"Saved {audio_path} ({waveform.shape[0]} samples, {waveform.dtype})")

        try:
            audio = read_audio(str(audio_path), sr=args.sr)
            print(f"read_audio -> {audio.shape[0]} samples")
        except Exception as exc:
            print("FAILED reading audio:", exc)
            results.append((duration, False, f"read_audio failed: {exc}"))
            continue

        try:
            batch = collator([{"audio": str(audio_path), "text": args.text, "prompt": ""}])
            input_ids_shape = batch["input_ids"].shape if "input_ids" in batch else None
            labels_shape = batch["labels"].shape if "labels" in batch else None
            print(f"collator succeeded: input_ids={input_ids_shape}, labels={labels_shape}")
            results.append((duration, True, f"collator OK input_ids={input_ids_shape}, labels={labels_shape}"))
        except Exception as exc:
            print("FAILED collating batch:", exc)
            results.append((duration, False, f"collator failed: {exc}"))

    print("\n=== Summary ===")
    for duration, ok, message in results:
        status = "PASS" if ok else "FAIL"
        print(f"{duration:.0f}s -> {status}: {message}")

    print(f"\nTemporary files are stored in {args.tmp_dir}")


if __name__ == "__main__":
    main()
