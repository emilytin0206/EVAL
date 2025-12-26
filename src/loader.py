import logging
from datasets import load_dataset

logger = logging.getLogger("MMLULoader")

class MMLUDataLoader:
    # MMLU 的完整 57 個子集列表
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

        # 處理 Subsets
        if not subsets:
            self.subsets = ['global_facts'] # 預設
        elif 'all' in subsets or subsets == ['all']:
            # 如果使用者指定 'all'，則展開為所有 57 個子集
            logger.info("Config 'all' detected. Expanding to all 57 MMLU subsets.")
            self.subsets = self.MMLU_SUBSETS
        else:
            self.subsets = subsets

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