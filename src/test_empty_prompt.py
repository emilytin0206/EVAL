import os
import sys
import json
import logging
import argparse

# 將專案根目錄加入路徑，確保可以引入 src 下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.client import OllamaClient
from src.loader import MMLUDataLoader
from src.scorer import Scorer

# 設定 Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BaselineTest")

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Empty Prompt Evaluation")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model name")
    # 預設測試的子集，可依需求調整
    parser.add_argument("--subsets", type=str, default="high_school_mathematics,high_school_world_history,high_school_physics,professional_law,business_ethics", help="Comma separated MMLU subsets")
    parser.add_argument("--limit", type=int, default=100, help="Num samples per subset (use 0 for all)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # 1. 初始化元件
    logger.info(f"🚀 Starting Baseline Evaluation (Empty Prompt) with Model: {args.model}")
    client = OllamaClient(model_name=args.model)
    subsets_list = [s.strip() for s in args.subsets.split(',')]
    
    # 使用 Loader 進行資料載入與切片
    loader = MMLUDataLoader(subsets=subsets_list, split=args.split, limit=args.limit)
    
    # 初始化 Scorer (使用 Q_begin 模式)
    scorer = Scorer(client, config_mode='Q_begin')

    # 2. 載入資料
    logger.info(f"Loading Datasets: {subsets_list}")
    dataset = loader.load_data()
    logger.info(f"Total samples loaded: {len(dataset)}")
    
    # 3. 定義空 Prompt
    # 這裡使用空字串，Scorer 會將其格式化為單純的 "Q: ... A:" 格式
    empty_prompt_text = "" 
    prompt_id = "baseline_empty_prompt"
    
    logger.info(f"Testing Empty Prompt...")

    # 4. 執行評測
    # 傳入 0 代表不再次抽樣，直接跑完 Loader 載入的所有資料
    res = scorer.score_instruction(empty_prompt_text, dataset, num_samples=0)
    
    result_data = {
        "score": res['score'],
        "prompt": "(Empty Prompt) - Baseline",
        "count": res['num_evals'],
        "id": prompt_id
    }
    
    logger.info(f"✅ Baseline Score: {res['score']:.2%}")

    # 5. 儲存結果
    os.makedirs(args.output_dir, exist_ok=True)
    # 檔名加上 baseline 標記
    out_filename = f"BASELINE_{args.model.replace(':', '-')}_{len(subsets_list)}subsets.json"
    out_path = os.path.join(args.output_dir, out_filename)
    
    final_output = {
        "source_file": "src/test_empty_prompt.py",
        "model": args.model,
        "subsets": subsets_list,
        "results": [result_data]
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()