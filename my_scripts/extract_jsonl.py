import json

# ===== 配置 =====
TEXT_FILE  = "/tsdata/minnanyu/cloud/wav_9/text"            # 原始文本文件
AUDIO_DIR  = "/asr-202011-202012"                             # 音频存放目录
JSONL_FILE = "output.jsonl"                                   # 输出 JSONL 文件名
NUM        = 100                                              # 提取前几条
# =================

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

count = 0
with open(JSONL_FILE, "w", encoding="utf-8") as fout:
    for line in lines[:NUM]:
        parts = line.split(None, 1)            # 只按第一个空格拆分
        if len(parts) < 2:
            continue

        audio_id = parts[0].strip()            # 如 15080004434_2147.64415
        text     = parts[1].strip()            # 如 凊采清采
        audio    = f"{AUDIO_DIR}/{audio_id}.wav"

        count += 1
        record = {"audio": audio, "text": text, "prompt": ""}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ 完成！共处理 {count} 条，JSONL 保存至 {JSONL_FILE}")
