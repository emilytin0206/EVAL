import os
import sys
import json
import glob
import logging
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.client import OllamaClient
from src.loader import MMLUDataLoader
from src.scorer import Scorer

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def main():
    parser = argparse.ArgumentParser(description="Automated Prompt Evaluation System")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")
    parser.add_argument("--subsets", type=str, default="global_facts", help="Comma separated MMLU subsets")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--prompt_dir", type=str, default="./prompts", help="Directory containing prompt JSONs")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--limit", type=int, default=5, help="Num samples per subset (use 0 for all)")
    
    args = parser.parse_args()
    
    # 1. 準備元件
    client = OllamaClient(model_name=args.model)
    subsets_list = [s.strip() for s in args.subsets.split(',')]
    
    # [修正 1] 將 limit 傳入 Loader，讓 Loader 負責每個子集取前 N 筆
    # 注意：請確保 src/loader.py 的 __init__ 已經更新接收 limit 參數
    loader = MMLUDataLoader(subsets=subsets_list, split=args.split, limit=args.limit)
    
    scorer = Scorer(client, config_mode='Q_begin')

    # 2. 載入資料
    logger.info("Loading Dataset...")
    dataset = loader.load_data()
    
    # 3. 確保目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 掃描檔案並執行
    json_files = glob.glob(os.path.join(args.prompt_dir, "*.json"))
    
    # [修正 2] 設定 Scorer 的 limit 邏輯
    # 因為 Loader 已經負責篩選資料了，所以這裡傳給 Scorer 0 (代表不抽樣，全跑)
    if args.limit > 0:
        logger.info(f"🔧 Config: Loader took first {args.limit} items per subset. Total loaded samples: {len(dataset)}")
        scorer_limit = 0 
    else:
        logger.info("🔧 Config: Limit set to 0. Running on FULL dataset.")
        scorer_limit = 0

    for json_file in json_files:
        full_file_name = os.path.basename(json_file)
        base_name = os.path.splitext(full_file_name)[0]
        
        logger.info(f"Processing: {full_file_name}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = []
        prompts = data.get("prompts", [])
        
        # [修正 3] 確保這裡的縮排正確，位於 for json_file 迴圈內部
        for idx, item in enumerate(prompts):
            # 兼容處理：無論輸入是字串還是物件，都取出 Prompt 文字
            p_text = item if isinstance(item, str) else item.get("text", "")
            
            # 雖然輸出不存 ID，但 Log 還是印一下方便你看進度
            p_id_log = f"p_{idx}" if isinstance(item, str) else item.get("id", f"p_{idx}")
            
            if not p_text: continue
            
            logger.info(f"Testing: {p_id_log}")
            
            # [修正 4] 傳入 scorer_limit (為 0，代表執行所有 dataset 內容)
            res = scorer.score_instruction(p_text, dataset, num_samples=scorer_limit)
            
            results.append({
                "score": res['score'],
                "prompt": p_text,
                "count": res['num_evals']
            })
            
            logger.info(f"Score: {res['score']:.2%}")

        # 輸出結果檔案
        out_filename = f"{base_name}_result.json"
        out_path = os.path.join(args.output_dir, out_filename)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_file": full_file_name,
                "model": args.model,
                "subsets": subsets_list,
                "results": results
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved results to: {out_filename}")

if __name__ == "__main__":
    main()