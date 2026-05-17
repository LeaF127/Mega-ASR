<p align="center">
  <img src="assets/mega_asr_logo.png" alt="Mega-ASR Logo" width="220">
</p>

<h1 align="center">Mega-ASR: Towards In-the-Wild Speech Recognition</h1>

<p align="center">
  <b>Robust Automatic Speech Recognition for Complex Real-World Acoustic Scenarios</b>
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/Mega-ASR/"><b>Homepage</b></a> |
  <a href="#model-download"><b>Model Download</b></a> |
  <a href="#installation"><b>Our Bench Download</b></a> |
  <a href="#paper"><b>Paper</b></a> |
</p>

<p align="center">
  <a href="https://xzf-thu.github.io/Mega-ASR/">
    <img src="https://img.shields.io/badge/Project-Homepage-purple">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange">
  <img src="https://img.shields.io/badge/ASR-Robust%20Speech%20Recognition-brightgreen">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green">
</p>

## Introduction

## Introduction

We present Mega-ASR, an open-source speech recognition model for robust ASR under complex dirty speech conditions. Mega-ASR is designed to improve recognition stability on medium- and high-error-rate audio, including noisy, far-field, distorted, stuttering, echoic, obstructed, and mixed-interference speech.

This repository contains the official implementation, model weights, core training data, and evaluation toolkit for Mega-ASR.

This repository is currently under active development.


## 🔥🔥🔥 News!!

- **May 20, 2025**: 🔥 We release **Mega-ASR**. Model weights on Hugging Face are coming soon.
- **May 20, 2025**: 🔥 We release **Voices-in-the-Wild-2M**, a benchmark for in-the-wild ASR robustness evaluation. [[Dataset]](https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-test-v2)
- **Coming soon**: 🔥 We will release the **DAPO-LoRA training code**.

## Highlights



## Model Download

We provide two Mega-ASR model variants for different usage scenarios.

| Model | Description | Download |
|---|---|---|
| **Mega-ASR for Dirty** | Optimized for dirty speech scenarios, including noisy, far-field, low-volume, degraded, and hard-to-recognize audio. | Coming soon |
| **Mega-ASR for All** | Built upon Mega-ASR for Dirty with a lightweight routing module that automatically distinguishes clean speech from degraded speech and selects the appropriate recognition path. | Coming soon |

After downloading the model weights, please specify the model path in the corresponding inference script or pass it through command-line arguments.


## Project Structure

```text
Mega-ASR/
├─ assets/
│  └─ Figures, logos, and other README resources.
│
├─ configs/
│  └─ Configuration files for SFT-LoRA and DAPO-LoRA training.
│
├─ data/
│  └─ Local data directory. Large-scale audio data is not tracked by Git.
│
├─ eval/
│  └─ evaluate_wer.py
│     WER/CER evaluation utilities for ASR robustness testing.
│
└─ src_MegaASR/
   ├─ inference/
   │  ├─ inference_MegaASR_for_dirty.py
   │  │  Dirty-speech inference without routing, designed for degraded audio.
   │  │
   │  └─ inference_MegaASR_for_all.py
   │     General inference with routing, supporting both dirty and general audio.
   │
   └─ train/
      ├─ SFT_lora/
      │  └─ SFT_lora.py
      │     SFT-LoRA training pipeline for acoustic robustness adaptation.
      │
      └─ DAPO_lora/
         └─ DAPO-LoRA training module, to be released in a future update.
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Mega-ASR.git
cd Mega-ASR
pip install -r requirements.txt
```

## Inference

### Inference for Dirty Audio

```bash
python src_MegaASR/inference/inference_MegaASR_for_dirty.py \
  --audio path/to/audio.wav \
  --model_path path/to/model
```

### Inference for General Audio

```bash
python src_MegaASR/inference/inference_MegaASR_for_all.py \
  --audio path/to/audio.wav \
  --model_path path/to/model
```

## Evaluation

```bash
python eval/evaluate_wer.py \
  --pred predictions.jsonl \
  --ref references.jsonl
```

## Training

### SFT-LoRA Training

```bash
python src_MegaASR/train/SFT_lora/SFT_lora.py \
  --config configs/sft_lora.yaml
```

### DAPO-LoRA Training

The DAPO-LoRA training module is under active research and will be released in a future version.

## Roadmap

- [x] Repository structure
- [ ] Inference for dirty audio
- [ ] Inference for general audio
- [ ] WER evaluation
- [ ] SFT-LoRA training
- [ ] Model checkpoint release
- [ ] DAPO-LoRA release

## Citation

If you find this project useful, please consider citing our work. Citation information will be updated after the release of the paper.

## License

This project will be released under the Apache-2.0 License.