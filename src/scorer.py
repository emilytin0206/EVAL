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

    def _get_normalized_prediction(self, prediction: str) -> str:
        """
        針對選擇題的標準化清理與答案提取 (Robust Version)。
        整合了 LaTeX 支援、關鍵字切割與智慧選取邏輯。
        """
        if not prediction: return ""
        text = str(prediction).strip()
        
        # 1. [最高優先] LaTeX Boxed 格式: \boxed{a} (忽略大小寫)
        if m := re.search(r'\\boxed\{\s*([a-e])\s*\}', text, re.IGNORECASE):
            return m.group(1).lower()

        # 2. [結論定位] 關鍵字切割
        # 尋找 "Answer is" 等詞，只保留其後的內容作為「結論區」
        text_lower = text.lower()
        keywords = ['answer is', 'answer:', 'the answer is', 'correct answer is', 'option:', 'choice:']
        
        conclusion_text = text_lower
        found_keyword = False
        
        for pat in keywords:
            if pat in text_lower:
                # 使用 rsplit 確保我們抓的是最後一次出現的關鍵字
                # 取 [-1] 代表取關鍵字「後面」的內容
                conclusion_text = text_lower.rsplit(pat, 1)[-1].strip()
                found_keyword = True
                break
        
        # 3. [提取選項] 根據是否鎖定結論區，決定抓頭還是抓尾
        
        # 3.1 尋找括號格式: (a), (b)
        matches_paren = re.findall(r'\(([a-e])\)', conclusion_text)
        if matches_paren:
            # 如果有鎖定結論區 -> 答案通常在開頭 -> 取第一個
            # 如果沒鎖定 (全文) -> 答案通常在結尾 (CoT) -> 取最後一個
            return matches_paren[0] if found_keyword else matches_paren[-1]
            
        # 3.2 尋找單獨字母: a, b (需有邊界 \b)
        matches_word = re.findall(r'\b([a-e])\b', conclusion_text)
        if matches_word:
            return matches_word[0] if found_keyword else matches_word[-1]

        # 4. [保底策略] 
        # 如果沒抓到明確選項，回傳清理後的結論文字 (移除句號)
        # 這能相容非選擇題的簡短回答 (如數字 "4")
        cleaned = conclusion_text
        return cleaned[:-1] if cleaned.endswith('.') else cleaned

    def _check_answer(self, prediction: str, target: str) -> float:
        pred_norm = self._get_normalized_prediction(str(prediction))
        target_norm = str(target).lower().strip()
        
        # 進行比對
        return 1.0 if pred_norm == target_norm else 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = None) -> dict:
        import random
        eval_data = dataset
        if num_samples and num_samples < len(dataset):
            # random.seed(42) # 可固定隨機
            eval_data = random.sample(dataset, num_samples)

        scores = []
        # 設定並發數，Ollama 本地端建議設為 1，避免記憶體爆掉
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for ex in eval_data:
                prompt = self._format_prompt(instruction, ex['input'])
                futures.append(executor.submit(self._run_single, prompt, ex['target']))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Eval", leave=False):
                res = future.result()
                if res is not None:  # <--- 只要不是 None (代表程式出錯)，就算是 0.0 也要加入！
                    scores.append(res)

        return {
            'score': float(np.mean(scores)) if scores else 0.0,
            'num_evals': len(scores)
        }

    def _run_single(self, prompt, target):
        try:
            pred = self.client.generate_text(prompt)
            return self._check_answer(pred, target)
        except Exception as e:
            logger.error(f"Error: {e}")
            return None