# Toji QLoRA GPU Quickstart (Colab)

## 1) Put bundle in an accessible path
Use this bundle from your local repo:

`dist/colab/toji_v2_colab_bundle_20260223T093203Z.tar.gz`

Expected SHA256:

`010f83b67e9273299eac07c8a01ae83604dc12e4cb84abeca95b59e9f3465a5d`

## 2) Start Colab with GPU runtime
In Colab: `Runtime -> Change runtime type -> GPU`.

Recommended locations:
- `/content/upload/` (runtime-local)
- Google Drive (if mount works in your runtime)
- Any absolute path, then set `TOJI_BUNDLE_PATH`

## 3) Run these cells

```bash
!nvidia-smi
```

```bash
!pip -q install -U pip
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers peft accelerate bitsandbytes safetensors sentencepiece datasets requests
```

```bash
import os, glob, hashlib, tarfile

EXPECTED_SHA256 = "010f83b67e9273299eac07c8a01ae83604dc12e4cb84abeca95b59e9f3465a5d"
EXTRACT_DIR = "/content/toji_bundle"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Optional explicit path override.
BUNDLE_PATH = os.environ.get("TOJI_BUNDLE_PATH", "").strip()

def _try_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return os.path.isdir("/content/drive/MyDrive")
    except Exception as e:
        print(f"Drive mount unavailable in this runtime: {e}")
        return False

patterns = []
if BUNDLE_PATH:
    patterns.append(BUNDLE_PATH)

if _try_mount_drive():
    patterns.append("/content/drive/MyDrive/**/*toji_v2_colab_bundle*.tar.gz")
else:
    print("Proceeding without drive.mount().")

patterns.extend([
    "/content/upload/toji_v2_colab_bundle*.tar.gz",
    "/content/**/*toji_v2_colab_bundle*.tar.gz",
])

candidates = []
for pattern in patterns:
    candidates.extend(glob.glob(pattern, recursive=True))

if not candidates:
    raise FileNotFoundError(
        "Bundle not found. Put tar.gz in /content/upload or MyDrive, "
        "or set TOJI_BUNDLE_PATH to full path."
    )

candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p))
bundle_path = candidates[-1]
print("Using:", bundle_path)

# Verify checksum before extract.
h = hashlib.sha256()
with open(bundle_path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
bundle_sha = h.hexdigest()
print("SHA256:", bundle_sha)
if EXPECTED_SHA256 and bundle_sha != EXPECTED_SHA256:
    raise ValueError(f"Checksum mismatch. expected={EXPECTED_SHA256} got={bundle_sha}")

with tarfile.open(bundle_path, "r:gz") as tar:
    tar.extractall(EXTRACT_DIR)
print("Extracted:", os.listdir(EXTRACT_DIR))
```

```bash
!mkdir -p /content/work/tools /content/work/data/toji_finetune
!cp /content/toji_bundle/repo_assets/train_toji_qlora.py /content/work/tools/
!cp /content/toji_bundle/repo_assets/eval_toji_qlora.py /content/work/tools/
!cp /content/toji_bundle/dataset/toji_v2_messages_train.jsonl /content/work/data/toji_finetune/
!cp /content/toji_bundle/dataset/toji_v2_messages_val.jsonl /content/work/data/toji_finetune/
!cp /content/toji_bundle/dataset/toji_v2_messages_test.jsonl /content/work/data/toji_finetune/
```

```bash
%cd /content/work
!mkdir -p /content/work/run_logs
!pkill -f "train_toji_qlora.py --run-name toji_qwen25_7b_colab_v1" || true
!nohup bash -lc 'PYTHONUNBUFFERED=1 PYTHONPATH=. python -u tools/train_toji_qlora.py \
  --train-file data/toji_finetune/toji_v2_messages_train.jsonl \
  --val-file data/toji_finetune/toji_v2_messages_val.jsonl \
  --run-name toji_qwen25_7b_colab_v1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --quantization qlora \
  --epochs 2 \
  --max-seq-len 1024 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --gradient-checkpointing \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --save-total-limit 3' > /content/work/run_logs/toji_qwen25_7b_colab_v1.log 2>&1 &
!tail -n 40 /content/work/run_logs/toji_qwen25_7b_colab_v1.log
```

```bash
# Progress check (run repeatedly in a separate cell)
!ps -ef | grep train_toji_qlora.py | grep -v grep
!ls -lah /content/work/outputs/toji_qlora/toji_qwen25_7b_colab_v1/checkpoints
!tail -n 60 /content/work/run_logs/toji_qwen25_7b_colab_v1.log
```

```bash
!PYTHONPATH=. python tools/eval_toji_qlora.py \
  --runtime huggingface \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path outputs/toji_qlora/toji_qwen25_7b_colab_v1/adapter \
  --test-file data/toji_finetune/toji_v2_messages_test.jsonl \
  --max-samples 50 \
  --out outputs/toji_eval/toji_qwen25_7b_colab_v1_eval.json
```

## 4) Download adapter artifacts

```bash
!tar -czf /content/toji_qwen25_7b_colab_v1_artifacts.tar.gz \
  -C /content/work/outputs/toji_qlora/toji_qwen25_7b_colab_v1 \
  adapter tokenizer training_plan.json training_summary.json
```

Then use `files.download("/content/toji_qwen25_7b_colab_v1_artifacts.tar.gz")`.
