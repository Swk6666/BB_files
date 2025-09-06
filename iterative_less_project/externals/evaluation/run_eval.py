# /root/iterative_less_project/externals/evaluation/run_eval.py
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F # <--- 新增导入

from .categories import categories, subcategories
from .utils import (dynamic_import_function, get_next_word_predictions,
                        load_hf_lm_and_tokenizer)

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    return " ".join(subject.split("_"))

def format_example(df, idx, include_answer=True):
    prompt = str(df.iloc[idx, 0])
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, data_to_eval, batch_size=1, k=5):
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    
    for i in range(data_to_eval.shape[0]):
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt_end = format_example(data_to_eval, i, include_answer=False)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if not prompt.endswith("\n"):
                prompt += "\n"
            prompt += "The answer is:"

        prompts.append(prompt)

    answer_choice_ids = [tokenizer.encode(" " + choice, add_special_tokens=False)[-1] for choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, batch_size=batch_size, return_token_predictions=False
    )

    cors = []
    groud_truths = data_to_eval.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/mmlu")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base model for LoRA adapters.")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--chat_formatting_function", type=str, default="externals.evaluation.templates.create_prompt_with_tulu_chat_format")
    parser.add_argument("--eval_valid", action="store_true", help="如果提供，则在验证集 (dev) 上进行评估。")

    args = parser.parse_args()

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        base_model_path=args.base_model_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        load_in_8bit=args.load_in_8bit,
        device_map="auto"
    )

    mmlu_data_dir = os.path.join(args.data_dir)
    
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(mmlu_data_dir, "test")) if "_test.csv" in f])
    if args.subjects:
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    eval_split_name = 'dev' if args.eval_valid else 'test'
    for subject in tqdm(subjects, desc=f"Evaluating on {eval_split_name} set"):
        dev_df = pd.read_csv(os.path.join(mmlu_data_dir, "dev", f"{subject}_dev.csv"), header=None)
        test_df = pd.read_csv(os.path.join(mmlu_data_dir, "test", f"{subject}_test.csv"), header=None)

        if args.eval_valid:
            data_to_eval = dev_df
            k_shots = 0
        else:
            data_to_eval = test_df
            k_shots = args.ntrain
        
        cors, acc, probs = eval_hf_model(
            args, subject, model, tokenizer, 
            dev_df, 
            data_to_eval,
            args.eval_batch_size, 
            k=k_shots
        )
        
        all_cors.append(cors)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories:
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        
        data_to_eval["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            data_to_eval[f"choice{choice}_probs"] = probs[:, j]
        data_to_eval.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors)) if all_cors else 0.0
    print(f"Average accuracy on {eval_split_name} set: {weighted_acc:.4f}")

    # --- 【代码修改开始】计算并记录平均损失 ---
    all_gts = []
    all_probs_list = []
    for subject in subjects:
        subject_df = pd.read_csv(os.path.join(args.save_dir, f"{subject}.csv"))
        all_gts.extend(subject_df.iloc[:, -1- (4 + 1)].values) # Ground truth is the column before 'correct' and 4 prob columns
        all_probs_list.append(subject_df[[f"choice{c}_probs" for c in choices]].values)

    all_probs_tensor = torch.tensor(np.concatenate(all_probs_list))
    
    label_to_idx = {label: i for i, label in enumerate(choices)}
    gts_indices = torch.tensor([label_to_idx[str(label)] for label in all_gts])

    # 使用负对数似然损失 (NLL Loss) 计算交叉熵
    # torch.log(tensor) 得到 log-probabilities
    average_loss = F.nll_loss(torch.log(all_probs_tensor + 1e-9), gts_indices).item()
    print(f"Average validation loss: {average_loss:.4f}")
    # --- 【代码修改结束】 ---

    results = {
        "average_acc": weighted_acc,
        "average_loss": average_loss, # <-- 添加损失值
        "subcat_acc": {
            subcat: np.mean(np.concatenate(subcat_cors[subcat]))
            for subcat in subcat_cors if subcat_cors[subcat]
        },
        "cat_acc": {
            cat: np.mean(np.concatenate(cat_cors[cat]))
            for cat in cat_cors if cat_cors[cat]
        },
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()