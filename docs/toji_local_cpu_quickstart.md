# Toji Local CPU Quickstart (No Colab)

This runs fully on your local machine using files already in this repo.

## 1) Install dependencies in local Python 3.12

```bash
python3 -m pip install -U transformers peft accelerate datasets sentencepiece safetensors requests
```

If `torch` is missing:

```bash
python3 -m pip install torch torchvision torchaudio
```

## 2) Plan-only sanity check (fast)

```bash
cd /Users/kaluarunsi/Desktop/autonomous-dispatcher/logistics_intelligence
PYTHONPATH=. python3 tools/train_toji_qlora.py \
  --train-file data/toji_finetune/toji_v2_messages_train.jsonl \
  --val-file data/toji_finetune/toji_v2_messages_val.jsonl \
  --run-name toji_qwen25_05b_cpu_v1 \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --quantization lora \
  --epochs 1 \
  --max-seq-len 256 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 10 \
  --save-steps 25 \
  --eval-steps 25 \
  --save-total-limit 2 \
  --max-train-samples 200 \
  --max-val-samples 50 \
  --plan-only
```

## 3) Start local CPU training

```bash
cd /Users/kaluarunsi/Desktop/autonomous-dispatcher/logistics_intelligence
mkdir -p outputs/toji_qlora logs
PYTHONPATH=. python3 tools/train_toji_qlora.py \
  --train-file data/toji_finetune/toji_v2_messages_train.jsonl \
  --val-file data/toji_finetune/toji_v2_messages_val.jsonl \
  --run-name toji_qwen25_05b_cpu_v1 \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --quantization lora \
  --epochs 1 \
  --max-seq-len 256 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 10 \
  --save-steps 25 \
  --eval-steps 25 \
  --save-total-limit 2 \
  --max-train-samples 200 \
  --max-val-samples 50 \
  > logs/toji_qwen25_05b_cpu_v1.log 2>&1
```

## 4) Evaluate adapter

```bash
cd /Users/kaluarunsi/Desktop/autonomous-dispatcher/logistics_intelligence
PYTHONPATH=. python3 tools/eval_toji_qlora.py \
  --runtime huggingface \
  --hf-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path outputs/toji_qlora/toji_qwen25_05b_cpu_v1/adapter \
  --test-file data/toji_finetune/toji_v2_messages_test.jsonl \
  --max-samples 20 \
  --out outputs/toji_eval/toji_qwen25_05b_cpu_v1_eval.json
```

## 5) Package artifacts

```bash
cd /Users/kaluarunsi/Desktop/autonomous-dispatcher/logistics_intelligence
tar -czf toji_qwen25_05b_cpu_v1_artifacts.tar.gz \
  -C outputs/toji_qlora/toji_qwen25_05b_cpu_v1 \
  adapter tokenizer training_plan.json training_summary.json
```

