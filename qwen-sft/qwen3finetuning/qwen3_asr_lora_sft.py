# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# LoRA fine-tuning script for Qwen3-ASR, adapted from the full-parameter
# fine-tuning script (qwen3_asr_sft.py).
#
# Usage:
#   python qwen3finetuning/qwen3_asr_lora_sft.py \
#       --model_path Qwen/Qwen3-ASR-1.7B \
#       --train_file train.jsonl \
#       --eval_file eval.jsonl \
#       --output_dir outputs/lora_sft \
#       --lora_scope encoder_aligner \
#       --lr 1e-5 --lr_encoder 5e-6 --lr_aligner 5e-6

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_asr import Qwen3ASRModel
from safetensors.torch import load_file as safe_load_file
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


# ---------------------------------------------------------------------------
# LoRA target modules  -- same as MegaASR A2S-SFT
# ---------------------------------------------------------------------------

LORA_TARGETS = {
    "encoder": r"^audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$",
    "aligner": r"^audio_tower\.(conv_out|proj1|proj2)$",
    "encoder_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "encoder_b4_aligner": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.(20|21|22|23)\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$)"
    ),
    "llm": r"^model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "all": (
        r"^(audio_tower\.(conv_out|proj1|proj2)$"
        r"|audio_tower\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
        r"|model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)"
    ),
}


# ---------------------------------------------------------------------------
# Forward patch (same as qwen3_asr_sft.py)
# ---------------------------------------------------------------------------

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ---------------------------------------------------------------------------
# 显存监控工具
# ---------------------------------------------------------------------------

def log_gpu_memory(stage: str, extra: str = ""):
    if not torch.cuda.is_available():
        return
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    peak_mb = torch.cuda.max_memory_reserved() / 1024**2
    message = f"[memory] {stage}: allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB peak={peak_mb:.1f}MB"
    if extra:
        message += f" | {extra}"
    print(message)


# ---------------------------------------------------------------------------
# Checkpoint utils (same as qwen3_asr_sft.py)
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def copy_hf_files(src_dir: str, dst_dir: str):
    """Copy tokenizer / config files so the checkpoint can be used standalone."""
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_hf_files(self.base_model_path, ckpt_dir)
        return control


# ---------------------------------------------------------------------------
# Audio & data helpers (same as qwen3_asr_sft.py)
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRLoRA:
    processor: Any
    sampling_rate: int = 16000
    min_duration: float = 0.5  # 最小音频时长（秒），过短会导致 audio_tower 报 IndexError

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 过滤过短的音频
        valid = []
        for f in features:
            dur = librosa.get_duration(path=f["audio"])
            if dur < self.min_duration:
                print(f"[跳过] 音频过短 ({dur:.1f}s < {self.min_duration}s): {f['audio']}")
                continue
            valid.append(f)

        if not valid:
            raise ValueError(
                f"batch 内所有音频均短于 {self.min_duration}s，"
                "请检查数据或调低 --min_duration"
            )

        audio_paths = [f["audio"] for f in valid]
        prefix_texts = [f["prefix_text"] for f in valid]
        targets = [f["target"] for f in valid]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ---------------------------------------------------------------------------
# LoRA Trainer
# ---------------------------------------------------------------------------

class Qwen3ASRLoRATrainer(Trainer):
    """Trainer for Qwen3-ASR LoRA fine-tuning.

    Features:
    - Convert input float tensors to model dtype (like CastFloatInputsTrainer)
    - Save only LoRA adapter weights (not full model)
    - Support loading adapter from safetensors for resume
    - Support per-component learning rates
    """

    def __init__(
        self,
        *args,
        processor=None,
        base_model_path: str = "",
        merged_from_lora_path: str = "",
        lr_encoder: float = 1e-5,
        lr_aligner: float = 1e-5,
        lr_llm: float = 1e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.base_model_path = base_model_path
        self.merged_from_lora_path = merged_from_lora_path
        self.lr_encoder = lr_encoder
        self.lr_aligner = lr_aligner
        self.lr_llm = lr_llm
        self._debug_batch_idx = 0

    # --- float type casting (same as CastFloatInputsTrainer) ---

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    # --- OOM 感知的 training_step（来自 MegaASR） ---

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._debug_batch_idx += 1
        if self._debug_batch_idx % 50 == 0 and torch.cuda.is_available():
            log_gpu_memory(f"batch={self._debug_batch_idx}")
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / 1024**2
                reserved_mb = torch.cuda.memory_reserved() / 1024**2
                print(
                    f"[memory] OOM at batch={self._debug_batch_idx} "
                    f"allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB",
                    file=__import__("sys").stderr,
                )
            raise

    # --- save only LoRA adapter ---

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapter weights (peft)
        self.model.thinker.save_pretrained(output_dir, safe_serialization=True)

        # Save processor (tokenizer + feature extractor)
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        # Record metadata
        self._write_text(output_dir, "base_model.txt", self.base_model_path)
        self._write_text(output_dir, "merged_from_lora.txt", self.merged_from_lora_path)

        # Remove full-model weight files saved by HF Trainer
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _write_text(output_dir: str, name: str, text: str):
        if text:
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                f.write(text + "\n")

    # --- resume from adapter.safetensors ---

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model or self.model
        adapter_path = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            model.thinker.load_state_dict(safe_load_file(adapter_path), strict=False)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    # --- per-component learning rates ---

    @staticmethod
    def _group_name(name: str) -> str:
        if "lora_" not in name:
            return "other"
        if any(x in name for x in [
            "audio_tower.conv_out", "audio_tower.proj1", "audio_tower.proj2"
        ]):
            return "aligner"
        if "audio_tower.layers." in name:
            return "encoder"
        if "model.layers." in name and "audio_tower.layers." not in name:
            return "llm"
        return "other"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        groups = {"encoder": [], "aligner": [], "llm": [], "other": []}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                groups[self._group_name(name)].append(param)

        lrs = {
            "encoder": self.lr_encoder,
            "aligner": self.lr_aligner,
            "llm": self.lr_llm,
            "other": self.args.learning_rate,
        }
        optim_groups = [
            {"params": params, "lr": lrs[name], "weight_decay": self.args.weight_decay}
            for name, params in groups.items()
            if params
        ]

        if self.args.process_index == 0:
            for name, params in groups.items():
                if params:
                    n_params = sum(p.numel() for p in params)
                    print(f"[optimizer] {name:7s}: {n_params:>10,} params  lr={lrs[name]:.2e}")

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR LoRA SFT")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-lora-sft-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--min_duration", type=float, default=0.5,
                    help="最小音频时长（秒），低于此值的音频将被跳过，默认 0.5")

    # LoRA
    p.add_argument("--lora_scope", type=str, default="encoder_aligner",
                   choices=list(LORA_TARGETS.keys()),
                   help="Which modules to apply LoRA to")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--merge_lora_into_base_from", type=str, default="",
                   help="Path to a previously trained LoRA adapter to merge "
                        "into the base model before training a new adapter")

    # Training hyper-params
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_encoder", type=float, default=None,
                   help="Overrides --lr for encoder LoRA params")
    p.add_argument("--lr_aligner", type=float, default=None,
                   help="Overrides --lr for aligner LoRA params")
    p.add_argument("--lr_llm", type=float, default=None,
                   help="Overrides --lr for LLM LoRA params")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=None, help="使用 bf16（Ampere+ GPU 推荐）")
    p.add_argument("--fp16", action="store_true", default=None, help="使用 fp16（V100 等旧 GPU）")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save / Resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0,
                   help="Automatically find and resume from the latest checkpoint")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    # --- 混合精度选择 ---
    bf16_enabled = args.bf16
    fp16_enabled = args.fp16
    if bf16_enabled is None and fp16_enabled is None:
        gpu_supports_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
        bf16_enabled = gpu_supports_bf16
        fp16_enabled = not gpu_supports_bf16
    elif bf16_enabled and fp16_enabled:
        raise ValueError("--bf16 和 --fp16 不能同时使用")
    else:
        bf16_enabled = bool(bf16_enabled)
        fp16_enabled = bool(fp16_enabled)

    model_dtype = torch.bfloat16 if bf16_enabled else torch.float16
    print(f"[precision] bf16={bf16_enabled} fp16={fp16_enabled} dtype={model_dtype}")

    # --- Load model ---
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # --- Optionally merge a previous LoRA adapter into the base model ---
    merge_path = args.merge_lora_into_base_from.strip()
    if merge_path:
        if args.resume or args.resume_from.strip():
            raise ValueError(
                "Do not use --merge_lora_into_base_from with --resume / --resume_from."
            )
        print(f"[merge_lora] Merging adapter from {merge_path} into base model ...")
        model.thinker = PeftModel.from_pretrained(
            model.thinker, merge_path, is_trainable=False
        ).merge_and_unload()
        print("[merge_lora] Merge complete.")

    # --- Apply LoRA ---
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS[args.lora_scope],
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    # 显存监控：只记录，不手动移 GPU（DDP / Trainer 会自动处理设备分配）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    log_gpu_memory("after_model_to_cuda")

    # 注意：PEFT + DDP + 梯度检查点三者不兼容，LoRA 本身已大幅降低显存，无需开启

    # --- Dataset ---
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"validation": args.eval_file} if args.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    # --- Data collator ---
    collator = DataCollatorForQwen3ASRLoRA(
        processor=processor, sampling_rate=args.sr, min_duration=args.min_duration
    )

    # --- Resolve per-component learning rates ---
    lr_encoder = args.lr_encoder if args.lr_encoder is not None else args.lr
    lr_aligner = args.lr_aligner if args.lr_aligner is not None else args.lr
    lr_llm = args.lr_llm if args.lr_llm is not None else args.lr

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory),
        dataloader_persistent_workers=bool(args.persistent_workers),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        ddp_find_unused_parameters=True,  # LoRA 下部分参数可能不参与梯度计算
        remove_unused_columns=False,
        report_to="none",
    )

    # --- Trainer ---
    trainer = Qwen3ASRLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
        processor=processor,
        base_model_path=args.model_path,
        merged_from_lora_path=merge_path,
        lr_encoder=lr_encoder,
        lr_aligner=lr_aligner,
        lr_llm=lr_llm,
    )

    # --- Resume ---
    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
