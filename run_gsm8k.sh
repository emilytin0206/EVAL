#!/bin/bash
set -e  # 若有任何錯誤立即停止

# 1. 設定變數
# 參數 1: 模型名稱 (預設 qwen2.5:7b)
MODEL_NAME=${1:-"llama2:7b"}  
# 參數 2: 任務類型 (預設 gsm8k，也可輸入 mmlu 子集)
# 
TASK_TYPE=${2:-"gsm8k"}

VENV_DIR="venv"
PROMPT_DIR="./prompts"
RESULT_DIR="./results"

# 測試樣本數 (0 代表跑全量)
#
LIMIT_SAMPLES=0

echo "🚀 Starting Evaluation Pipeline"
echo "🤖 Model: $MODEL_NAME"
echo "📋 Task/Subsets: $TASK_TYPE"
echo "ℹ️  Running in current Python environment."

# 2. 安裝/檢查依賴
#
echo "⬇️  Checking dependencies..."
pip install -q -r requirements.txt

# 3. 建立必要的資料夾
#
mkdir -p $PROMPT_DIR
mkdir -p $RESULT_DIR

# 5. 執行 Python 主程式
# 將 $TASK_TYPE 傳入 --subsets 參數
#
echo "🔥 Running Evaluation..."
python src/main.py \
    --model "$MODEL_NAME" \
    --subsets "$TASK_TYPE" \
    --prompt_dir "$PROMPT_DIR" \
    --output_dir "$RESULT_DIR" \
    --limit $LIMIT_SAMPLES

echo "✅ All done! Check results in $RESULT_DIR"