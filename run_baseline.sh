#!/bin/bash
set -e  # 若有任何錯誤立即停止

# 1. 設定變數
# 預設使用 qwen2.5:7b，執行時可傳入參數覆蓋 (例如: ./run_baseline.sh llama3)
MODEL_NAME=${1:-"qwen2.5:7b"}  
RESULT_DIR="./results"

# 設定要測試的 MMLU 子集 (保持與 run_eval.sh 一致，以便公平比較)
# SUBSETS="high_school_mathematics","high_school_world_history","high_school_physics","professional_law","business_ethics"
# SUBSETS=all
# SUBSETS="high_school_mathematics,high_school_world_history,high_school_macroeconomics,high_school_physics,business_ethics" 
# SUBSETS="high_school_mathematics" 
SUBSETS="high_school_world_history"
# 測試樣本數 (設為 0 代表跑全量，預設設為 100 與您原本設定一致)
LIMIT_SAMPLES=0

echo "🚀 Starting Baseline (Empty Prompt) Evaluation using Model: $MODEL_NAME"
echo "ℹ️  Running in current Python environment."

# 2. 安裝/檢查依賴 (確保環境一致)
echo "⬇️  Checking dependencies..."
pip install -q -r requirements.txt

# 3. 建立必要的資料夾
mkdir -p $RESULT_DIR

# 4. 執行 Python Baseline 程式
# 呼叫我們剛剛建立的 test_empty_prompt.py
echo "🔥 Running Baseline Evaluation..."
python src/test_empty_prompt.py \
    --model "$MODEL_NAME" \
    --subsets "$SUBSETS" \
    --output_dir "$RESULT_DIR" \
    --limit $LIMIT_SAMPLES

echo "✅ Baseline test done! Check results in $RESULT_DIR"