import os
import sys
import json
import glob
import logging
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.client import OllamaClient
from src.loader import MMLUDataLoader, GSM8KDataLoader # 引入新 Loader
from src.scorer import Scorer

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def main():
    parser = argparse.ArgumentParser(description="Automated Prompt Evaluation System")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")
    parser.add_argument("--subsets", type=str, default="global_facts", help="MMLU subsets or 'gsm8k'")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--prompt_dir", type=str, default="./prompts", help="Directory containing prompt JSONs")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--limit", type=int, default=5, help="Num samples per subset")
    
    args = parser.parse_args()
    
    # 1. 準備元件
    client = OllamaClient(model_name=args.model)
    subsets_list = [s.strip() for s in args.subsets.split(',')]
    
    # --- 關鍵修改：判斷任務類型 ---
    if args.subsets.lower() == 'gsm8k':
        logger.info("Task detected: GSM8K")
        loader = GSM8KDataLoader(split='test' if args.split=='validation' else args.split, limit=args.limit)
        task_type = 'gsm8k'
    else:
        logger.info(f"Task detected: MMLU ({args.subsets})")
        loader = MMLUDataLoader(subsets=subsets_list, split=args.split, limit=args.limit)
        task_type = 'mmlu'
    
    scorer = Scorer(client, config_mode='Q_begin')

    # 2. 載入資料
    logger.info("Loading Dataset...")
    dataset = loader.load_data()
    
    os.makedirs(args.output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(args.prompt_dir, "*.json"))
    
    scorer_limit = 0 # Loader 已處理 limit

    for json_file in json_files:
        full_file_name = os.path.basename(json_file)
        base_name = os.path.splitext(full_file_name)[0]
        logger.info(f"Processing: {full_file_name}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = []
        prompts = data.get("prompts", [])
        
        for idx, item in enumerate(prompts):
            p_text = item if isinstance(item, str) else item.get("text", "")
            p_id_log = f"p_{idx}" if isinstance(item, str) else item.get("id", f"p_{idx}")
            if not p_text: continue
            
            logger.info(f"Testing: {p_id_log}")
            
            # --- 關鍵修改：傳入 task_type ---
            res = scorer.score_instruction(p_text, dataset, num_samples=scorer_limit, task_type=task_type)
            
            results.append({
                "score": res['score'],
                "prompt": p_text,
                "count": res['num_evals']
            })
            logger.info(f"Score: {res['score']:.2%}")

        out_filename = f"{base_name}_result.json"
        out_path = os.path.join(args.output_dir, out_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_file": full_file_name,
                "model": args.model,
                "task_type": task_type,
                "subsets": subsets_list,
                "results": results
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved results to: {out_filename}")

if __name__ == "__main__":
    main()