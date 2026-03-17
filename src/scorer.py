import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger("Scorer")

class Scorer:
    def __init__(self, client, config_mode='Q_begin'):
        self.client = client
        self.instruction_pos = config_mode  # 強制設定 Prompt 結構

    def _format_prompt(self, instruction: str, question: str) -> str:
        # 依據你的要求，這裡實作 Q_begin 邏輯
        if self.instruction_pos == 'Q_begin':
            return f"{instruction}\n\nQ: {question}\nA:"
        return f"{instruction}\n{question}"

    def _get_normalized_prediction(self, prediction: str, task_type='mmlu') -> str:
        if not prediction: return ""
        text = str(prediction).strip()

        # 如果是 MMLU，保留您原有的 [a-e] 提取邏輯
        if task_type == 'mmlu':
            if m := re.search(r'\\boxed\{\s*([a-e])\s*\}', text, re.IGNORECASE):
                return m.group(1).lower()
            # ... (中間保留您原本的關鍵字與括號處理) ...
            matches_word = re.findall(r'\b([a-e])\b', text.lower())
            return matches_word[-1] if matches_word else text.lower()

        # 如果是 GSM8K，改用數字提取邏輯
        elif task_type == 'gsm8k':
            # 優先找 \boxed{數字}
            if m := re.search(r'\\boxed\{([\d\.,]+)\}', text):
                return m.group(1).replace(',', '')
            
            # 找 "The answer is X" 後面的數字
            if "answer is" in text.lower():
                suffix = text.lower().rsplit("answer is", 1)[-1]
                nums = re.findall(r'[\d\.,]+', suffix)
                if nums: return nums[0].replace(',', '')
            
            # 保底：抓全文最後一個出現的數字
            nums = re.findall(r'[\d\.,]+', text)
            return nums[-1].replace(',', '') if nums else ""

        return text
    def _check_answer(self, prediction: str, target: str) -> float:
        pred_norm = self._get_normalized_prediction(str(prediction))
        target_norm = str(target).lower().strip()
        
        # 進行比對
        return 1.0 if pred_norm == target_norm else 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = None, task_type='mmlu') -> dict:
        import random
        eval_data = dataset
        if num_samples and num_samples < len(dataset):
            eval_data = random.sample(dataset, num_samples)

        scores = []
        # 設定並發數，Ollama 本地端建議設為 1，避免記憶體爆掉
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for ex in eval_data:
                prompt = self._format_prompt(instruction, ex['input'])
                # 2. 修改此處：將 task_type 傳遞給 _run_single
                futures.append(executor.submit(self._run_single, prompt, ex['target'], task_type))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Eval", leave=False):
                res = future.result()
                if res is not None:
                    scores.append(res)

        return {
            'score': float(np.mean(scores)) if scores else 0.0,
            'num_evals': len(scores)
        }

    def _run_single(self, prompt, target, task_type):
        try:
            pred = self.client.generate_text(prompt)
            # 4. 修改此處：傳入 task_type 給 _get_normalized_prediction
            pred_norm = self._get_normalized_prediction(pred, task_type=task_type)
            target_norm = str(target).lower().strip()
            
            return 1.0 if pred_norm == target_norm else 0.0
        except Exception as e:
            logger.error(f"Error: {e}")
            return None