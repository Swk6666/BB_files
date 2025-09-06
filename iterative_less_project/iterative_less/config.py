# /root/iterative_less_project/iterative_less/config.py
import os

RANDOM_SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 模型源路径 ---
BASE_MODEL_NAME = "/root/autodl-tmp/modelscope_cache/modelscope/Llama-2-7b-ms"
# --- 实验输出路径 ---
DATADISK_PATH = "/root/autodl-tmp"
EXPERIMENT_NAME = "IterativeLess_Llama-7B_Final_Run_Robust"
RESULTS_BASE_DIR = os.path.join(DATADISK_PATH, "results")
ITERATIVE_OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)

# --- 项目内数据文件路径 ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WARMUP_POOL_FILENAME = os.path.join(DATA_DIR, "warmup_pool.jsonl")
CANDIDATE_POOL_FILENAME = os.path.join(DATA_DIR, "candidate_pool.jsonl")
MMLU_DATA_DIR = os.path.join(DATA_DIR, "mmlu")

# --- 实验参数 ---
NUM_ITERATIONS = 6
SELECTION_SIZE_PER_ITERATION = 150

GRADIENT_PROJECTION_DIM = 8192
GRADIENT_TYPE_VALIDATION = "sgd"
GRADIENT_TYPE_TRAIN_POOL_ADAM = "adam"

# --- 训练参数 ---
LORA_TRAIN_ARGS = {
    "num_train_epochs": 4,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bf16": True,
    "logging_steps": 10,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
}

# --- 评估参数 ---
# --- [最终OOM修复] ---
EVAL_ARGS = {
    "ntrain": 5,
    "eval_batch_size": 4,  # 从 8 进一步降低到 4
    "use_chat_format": True,
}
# --- [修复结束] ---