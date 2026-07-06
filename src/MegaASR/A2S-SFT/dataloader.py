# coding=utf-8
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from datasets import load_dataset


def read_audio(path: str, sr: int = 16000):
    if not path:
        raise ValueError("audio path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"audio file does not exist: {path}")
    audio, _ = librosa.load(path, sr=sr, mono=True)
    if audio is None or audio.size == 0:
        raise ValueError(f"audio file is empty or unreadable: {path}")
    return audio


def audio_messages(prompt: str):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]


@dataclass
class Qwen3ASRCollator:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_items = []
        for feature in features:
            try:
                audio = read_audio(feature["audio"], self.sampling_rate)
            except Exception:
                continue
            valid_items.append((feature, audio))

        if not valid_items:
            raise ValueError("No valid audio samples in batch")

        prompts = [x.get("prompt", "") for x, _ in valid_items]
        targets = [x["text"] for x, _ in valid_items]
        audios = [audio for _, audio in valid_items]

        prefixes = [
            self.processor.apply_chat_template(
                [audio_messages(p)],
                add_generation_prompt=True,
                tokenize=False,
            )[0]
            for p in prompts
        ]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefixes, targets)]

        batch = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_batch = self.processor(
            text=prefixes,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lens = prefix_batch["attention_mask"].sum(dim=1)
        full_lens = batch["attention_mask"].sum(dim=1)

        seq_len = labels.size(1)
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        for i, prefix_len in enumerate(prefix_lens):
            start = seq_len - int(full_lens[i]) if padding_side == "left" else 0
            labels[i, start:start + int(prefix_len)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels
        batch["__debug_info__"] = {
            "batch_size": len(valid_items),
            "samples": [
                {
                    "index": idx,
                    "audio": feature.get("audio", ""),
                    "audio_length_s": round(len(audio) / self.sampling_rate, 4),
                    "text_length": len(target),
                    "text_preview": target[:80],
                }
                for idx, ((feature, audio), target) in enumerate(zip(valid_items, targets))
            ],
        }
        return batch


def build_datasets(train_file: str, eval_file: str = ""):
    files = {"train": train_file}
    if eval_file:
        files["validation"] = eval_file
    return load_dataset("json", data_files=files)
