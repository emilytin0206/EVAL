import logging
from datasets import load_dataset

logger = logging.getLogger("MMLULoader")

class MMLUDataLoader:
    # MMLU 的完整 57 個子集列表
    CATEGORIES = {
       "stem": [
            "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics", "computer_security",
            "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
            "high_school_physics", "machine_learning", "statistics"
        ],
        "humanities": [
            "formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history",
            "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
            "philosophy", "prehistory", "professional_law", "world_religions"
        ],
        "social_sciences": [
            "econometrics", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
            "human_sexuality", "professional_psychology", "public_relations", "sociology"
        ],
        "other": [
            "business_ethics", "clinical_knowledge", "college_medicine", "dietetics", "global_facts",
            "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
            "professional_accounting", "professional_medicine", "security_studies", "us_foreign_policy", "virology"
        ]
    }

    MMLU_SUBSETS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    def __init__(self, subsets=None, split='validation', limit=0):  
        self.split = split
        self.limit = limit
        self.choices_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        if not subsets:
            self.subsets = ['global_facts']
        elif 'all' in subsets or subsets == ['all']:
            self.subsets = self.MMLU_SUBSETS
        else:
            # 2. 核心邏輯：動態解析領域與單獨子集
            final_subsets = []
            for s in subsets:
                if s in self.CATEGORIES:
                    # 如果是領域關鍵字，展開該領域下所有子集
                    logger.info(f"Category '{s}' detected. Expanding to {len(self.CATEGORIES[s])} subsets.")
                    final_subsets.extend(self.CATEGORIES[s])
                else:
                    # 如果不是領域名稱，則視為單獨的子集名稱 (例如 'abstract_algebra')
                    final_subsets.append(s)
            
            # 使用 set 去重，避免同時輸入領域又輸入該領域內子集導致重複載入
            self.subsets = list(dict.fromkeys(final_subsets))

    def format_mmlu_example(self, example):
        """將 MMLU 格式轉換為 Q_begin 需要的純文字輸入"""
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        
        formatted_choices = [f"({l}) {c}" for l, c in zip(['A','B','C','D'], choices)]
        full_input = f"{question}\n" + "\n".join(formatted_choices)
        
        return {
            'input': full_input,
            'target': self.choices_map[answer_idx]
        }

    def load_data(self):
        combined_data = []
        for subset in self.subsets:
            logger.info(f"Loading subset: {subset}...")
            try:
                ds = load_dataset("cais/mmlu", subset, split=self.split)
                subset_data = [self.format_mmlu_example(ex) for ex in ds]
                
                # 每個子集單獨切片
                if self.limit > 0:
                    subset_data = subset_data[:self.limit]
                    logger.info(f"  -> Loaded {len(subset_data)} samples from {subset}")
                
                combined_data.extend(subset_data)
            except Exception as e:
                logger.error(f"Failed to load {subset}: {e}")
        return combined_data



class GSM8KDataLoader:
    def __init__(self, split='test', limit=0):
        self.split = split
        self.limit = limit

    def format_gsm8k_example(self, example):
        """提取問題與 #### 後的數字答案"""
        question = example['question']
        # GSM8K 答案格式為 "解題過程... #### 數字"
        full_answer = example['answer']
        target = full_answer.split('####')[-1].strip().replace(',', '')
        
        return {
            'input': question,
            'target': target
        }

    def load_data(self):
        from datasets import load_dataset
        logger.info(f"Loading GSM8K dataset (split: {self.split})...")
        try:
            ds = load_dataset("openai/gsm8k", "main", split=self.split)
            data = [self.format_gsm8k_example(ex) for ex in ds]
            if self.limit > 0:
                data = data[:self.limit]
            return data
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []