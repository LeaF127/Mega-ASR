#!/usr/bin/env python3
import argparse
import contextlib
import os
import re
import wave

TIMESTAMP_LINE_RE = re.compile(
    r'^\s*(?:Speaker\s+\d+\s*)?(\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)',
    re.IGNORECASE,
)


def parse_timestamp(timestamp_text):
    normalized = timestamp_text.replace("，", ",").replace(",", ".")
    parts = normalized.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp format: {timestamp_text}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_timestamp(seconds):
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{ms:03d}"


def extract_timestamps(transcript_path):
    timestamps = []
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = TIMESTAMP_LINE_RE.match(line)
            if not match:
                continue
            try:
                ts = parse_timestamp(match.group(1))
            except ValueError:
                continue
            timestamps.append(ts)
    timestamps = sorted(set(timestamps))
    return [ts for ts in timestamps if ts >= 0.0]


def get_wav_duration(wav_path):
    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def write_wav_segment(src_wav_path, output_path, start_sec, end_sec):
    with contextlib.closing(wave.open(src_wav_path, "rb")) as src, contextlib.closing(
        wave.open(output_path, "wb")
    ) as dst:
        params = src.getparams()
        rate = src.getframerate()
        start_frame = int(round(start_sec * rate))
        end_frame = int(round(end_sec * rate))
        if start_frame < 0:
            start_frame = 0
        if end_frame > src.getnframes():
            end_frame = src.getnframes()
        num_frames = max(0, end_frame - start_frame)
        if num_frames == 0:
            raise ValueError(
                f"Segment duration is zero for {output_path}: {start_sec} - {end_sec}"
            )
        dst.setparams(params)
        src.setpos(start_frame)
        frames = src.readframes(num_frames)
        dst.writeframes(frames)


def split_wav_by_timestamps(src_wav_path, timestamps, output_dir, include_last, min_duration):
    if not timestamps:
        return []

    duration = get_wav_duration(src_wav_path)
    if include_last and timestamps[-1] < duration - 1e-6:
        timestamps = timestamps + [duration]

    segments = []
    for idx in range(len(timestamps) - 1):
        start = timestamps[idx]
        end = timestamps[idx + 1]
        if end - start < min_duration:
            continue
        start_tag = format_timestamp(start)
        end_tag = format_timestamp(end)
        output_name = f"{start_tag}_{end_tag}.wav"
        output_path = os.path.join(output_dir, output_name)
        write_wav_segment(src_wav_path, output_path, start, end)
        segments.append(output_path)
    return segments


def find_transcript_files(root, keep_dir_name):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == keep_dir_name and dirpath == os.path.join(root, keep_dir_name):
            continue
        if "transcript.txt" in filenames or "transript.txt" in filenames:
            results.append(os.path.join(dirpath, "transcript.txt") if "transcript.txt" in filenames else os.path.join(dirpath, "transript.txt"))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="根据 kept_folders 下 transcript.txt 中的时间戳切分对应 wav/xxx.wav，并输出到 splited_wavs/xxx/。"
    )
    parser.add_argument("root", nargs="?", default=".", help="工作根目录，默认当前目录")
    parser.add_argument("--keep-dir", default="kept_folders", help="保存 transcript 的目录")
    parser.add_argument("--wav-dir", default="wav", help="音频文件目录")
    parser.add_argument("--out-dir", default="splited_wavs", help="切分后音频输出目录")
    parser.add_argument("--min-duration", type=float, default=0.05, help="最小切分长度，单位秒")
    parser.add_argument(
        "--no-last-segment",
        action="store_true",
        help="不将最后一个时间点到音频末尾作为片段输出",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    wav_root = os.path.join(root, args.wav_dir)
    output_root = os.path.join(root, args.out_dir)
    keep_root = os.path.join(root, args.keep_dir)

    if not os.path.isdir(keep_root):
        raise FileNotFoundError(f"保留目录不存在: {keep_root}")
    if not os.path.isdir(wav_root):
        raise FileNotFoundError(f"wav 目录不存在: {wav_root}")

    os.makedirs(output_root, exist_ok=True)

    transcripts = find_transcript_files(keep_root, args.keep_dir)
    if not transcripts:
        print("未找到任何 transcript.txt 文件。")
        return

    total = 0
    skipped = 0
    for transcript_path in transcripts:
        total += 1
        folder_name = os.path.basename(os.path.dirname(transcript_path))
        wav_path = os.path.join(wav_root, f"{folder_name}.wav")
        if not os.path.isfile(wav_path):
            print(f"跳过: 未找到对应 wav 文件: {wav_path}")
            skipped += 1
            continue

        timestamps = extract_timestamps(transcript_path)
        if not timestamps:
            print(f"跳过: 未解析到时间戳: {transcript_path}")
            skipped += 1
            continue

        out_folder = os.path.join(output_root, folder_name)
        os.makedirs(out_folder, exist_ok=True)
        segments = split_wav_by_timestamps(
            wav_path,
            timestamps,
            out_folder,
            include_last=not args.no_last_segment,
            min_duration=args.min_duration,
        )

        if segments:
            print(f"切分完成: {folder_name} -> {len(segments)} 片段, 输出: {out_folder}")
        else:
            print(f"未生成片段: {folder_name} (可能所有片段小于最小长度)")

    print(f"处理完成: {total} 个 transcript，跳过 {skipped} 个")


if __name__ == "__main__":
    main()
