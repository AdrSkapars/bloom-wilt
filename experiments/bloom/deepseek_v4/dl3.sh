#!/bin/bash
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731", max_workers=8)
print("DOWNLOAD DONE", p, flush=True)
EOF
echo "EXIT=$?"
