import json
import os

# ===== 配置 =====
TEXT_FILE  = "/tsdata/minnanyu/cloud/wav_9/text"            # 原始文本文件
AUDIO_DIR  = "/asr-202011-202012"                             # 音频存放目录
OUTPUT_DIR = "./jsonl"                                        # 文件输出目录
TRAIN_FILE = "train.jsonl"                                    # 训练集输出
TEST_FILE  = "test.jsonl"                                     # 测试集输出
TRAIN_NUM  = 100                                              # 训练集条数
TEST_NUM   = 10                                               # 测试集条数
# =================

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

train_lines = lines[:TRAIN_NUM]
test_lines  = lines[TRAIN_NUM:TRAIN_NUM + TEST_NUM]

def process(lines_subset, out_file, tag):
    count = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    out_file = os.path.join(OUTPUT_DIR, out_file)
    
    with open(out_file, "w", encoding="utf-8") as fout:
        for line in lines_subset:
            parts = line.split(None, 1)        # 只按第一个空格拆分
            if len(parts) < 2:
                continue

            audio_id = parts[0].strip()        # 如 15080004434_2147.64415
            text     = parts[1].strip()        # 如 凊采清采
            audio    = f"{AUDIO_DIR}/{audio_id}.wav"

            count += 1
            record = {"audio": audio, "text": text, "prompt": ""}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ {tag}完成！共 {count} 条，保存至 {out_file}")

process(train_lines, TRAIN_FILE, "训练集")
process(test_lines,  TEST_FILE,  "测试集")