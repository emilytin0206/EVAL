import re
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger("Scorer")

class Scorer:
    def __init__(self, client, config_mode='Q_begin'):
        self.client = client
        self.instruction_pos = config_mode  # 強制設定 Prompt 結構

    def _format_prompt(self, instruction: str, question: str, task_type='mmlu') -> str:
        """
        將評測 Instruction 與問題結合，並【強制注入輸出格式要求】(策略 A)。
        """
        # 根據任務類型加上對應的輸出限制
        if task_type == 'mmlu':
            format_req = "Provide your step-by-step reasoning, and format your final answer exactly as <answer>X</answer> where X is A, B, C, D, or E."
        elif task_type == 'gsm8k':
            format_req = "Provide your step-by-step reasoning, and format your final numeric answer exactly as <answer>X</answer> where X is just the number."
        else:
            format_req = ""

        # 結合原有的 instruction (如果有) 與 format_req
        if instruction:
            final_instruction = f"{instruction}\n{format_req}"
        else:
            final_instruction = format_req  # 適用於 Baseline 空 Prompt 測試

        if self.instruction_pos == 'Q_begin':
            return f"{final_instruction}\n\nQ: {question}\nA:"
        
        return f"{final_instruction}\n{question}"

    def _get_normalized_prediction(self, prediction: str, task_type='mmlu') -> str:
        """
        從模型的回答中萃取出最終答案。優先尋找 <answer> 標籤，若無則啟用強健的保底機制。
        """
        if not prediction: return ""
        text = str(prediction).strip()

        # ==========================================
        # 任務一：MMLU (選擇題)
        # ==========================================
        if task_type == 'mmlu':
            # 1. [最高優先 - 策略 A] 尋找 <answer>X</answer> 標籤
            if m := re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', text):
                return m.group(1).lower()

            # 2. [保底 - 策略 B] 萬一模型沒照做，啟用嚴謹的字串萃取
            # 2-A. 尋找 \boxed 格式
            if m := re.search(r'\\boxed\{\s*([A-Ea-e])\s*\}', text):
                return m.group(1).lower()

            # 2-B. 定位結論區塊 (找結論關鍵字，並擷取之後的內容)
            conclusion_text = text
            keywords = ['(?i)answer is', '(?i)answer:', '(?i)option:', '(?i)choice:']
            for pat in keywords:
                matches = list(re.finditer(pat, conclusion_text))
                if matches:
                    last_match = matches[-1]
                    conclusion_text = conclusion_text[last_match.end():]
                    break

            # 2-C. 在結論區尋找被明確標點包覆的選項: (A), [A], A., Option A
            strict_patterns = [
                r'[\(\[]\s*([A-Ea-e])\s*[\)\]]',          # (A) 或 [A]
                r'\b([A-Ea-e])\s*[\.\)]',                 # A. 或 A)
                r'(?i)\b(?:Option|Choice)\s+([A-Ea-e])\b' # Option A
            ]
            for pattern in strict_patterns:
                if m := re.search(pattern, conclusion_text):
                    return m.group(1).lower()

            # 2-D. 最後手段：只抓大寫字母 A-E (避免抓到小寫冠詞 a dog)
            if m := re.search(r'\b([A-E])\b', conclusion_text):
                return m.group(1).lower()

            return ""

        # ==========================================
        # 任務二：GSM8K (數學計算題)
        # ==========================================
        elif task_type == 'gsm8k':
            # 1. [最高優先 - 策略 A] 尋找 <answer>數字</answer>
            if m := re.search(r'<answer>\s*([\d\.,]+)\s*</answer>', text):
                return m.group(1).replace(',', '')

            # 2. [保底] 舊版 GSM8K 數字萃取
            if m := re.search(r'\\boxed\{([\d\.,]+)\}', text):
                return m.group(1).replace(',', '')
            
            if "answer is" in text.lower():
                suffix = text.lower().rsplit("answer is", 1)[-1]
                nums = re.findall(r'[\d\.,]+', suffix)
                if nums: return nums[0].replace(',', '')
            
            # 抓全文最後一個數字
            nums = re.findall(r'[\d\.,]+', text)
            return nums[-1].replace(',', '') if nums else ""

        return text

    def _check_answer(self, prediction: str, target: str, task_type: str = 'mmlu') -> float:
        """
        統一的比對邏輯：取得萃取後的答案並與標準答案比對。
        """
        pred_norm = self._get_normalized_prediction(str(prediction), task_type=task_type)
        target_norm = str(target).lower().strip()
        
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
                # 在這裡將 task_type 傳給 _format_prompt
                prompt = self._format_prompt(instruction, ex['input'], task_type=task_type)
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
            # 取得模型回答
            pred = self.client.generate_text(prompt)
            # 交給 _check_answer 進行清洗與判定
            return self._check_answer(pred, target, task_type=task_type)
        except Exception as e:
            logger.error(f"Error: {e}")
            return None