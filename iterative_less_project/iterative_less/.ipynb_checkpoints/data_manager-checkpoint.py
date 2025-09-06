# /root/iterative_less_project/iterative_less/data_manager.py
import json
import os
from datasets import load_dataset, Dataset
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, candidate_pool_path: str):
        if not os.path.exists(candidate_pool_path):
            raise FileNotFoundError(f"候选数据池文件未找到: {candidate_pool_path}")
        
        logger.info(f"正在从 {candidate_pool_path} 加载候选数据池...")
        self.all_data = load_dataset("json", data_files=candidate_pool_path, split="train")
        
        if "original_index" not in self.all_data.column_names:
            logger.warning("数据中未找到 'original_index' 列，将自动创建。")
            self.all_data = self.all_data.add_column("original_index", range(len(self.all_data)))

        self.num_total_samples = len(self.all_data)
        logger.info(f"成功加载 {self.num_total_samples} 条候选数据。")

    def get_samples_by_indices(self, indices: list[int]) -> Dataset:
        """根据索引列表从数据集中选择样本。"""
        # datasets库的select方法非常高效
        return self.all_data.select(indices)

    @staticmethod
    def save_dataset_to_jsonl(dataset: Dataset, file_path: str):
        """将Hugging Face数据集对象保存为jsonl文件。"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset.to_json(file_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"成功保存 {len(dataset)} 条数据到 {file_path}")