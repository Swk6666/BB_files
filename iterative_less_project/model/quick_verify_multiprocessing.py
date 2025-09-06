# /root/iterative_less_project/quick_verify_multiprocessing.py
import os
import json
import sys
from transformers import AutoTokenizer

# Add project root to path to find the module
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from externals.less.data_selection.get_training_dataset import get_training_dataset
from iterative_less.config import BASE_MODEL_NAME, CANDIDATE_POOL_FILENAME

def run_verification():
    print("="*50)
    print("🚀 Starting Quick Verification for Deadlock Fix 🚀")
    print("="*50)

    # 1. Create a small dummy data file to test with
    dummy_file_path = "dummy_data.jsonl"
    print(f"1. Creating a small dummy data file: {dummy_file_path}")
    with open(dummy_file_path, "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"This is test sentence number {i}."}) + "\n")
    
    # 2. Load the tokenizer (doesn't need GPU)
    print(f"2. Loading tokenizer from: {BASE_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load tokenizer. Please ensure the path in config.py is correct.")
        print(f"   Error details: {e}")
        os.remove(dummy_file_path)
        return

    # 3. Call the target function with num_workers=1
    print("\n3. Calling get_training_dataset with num_workers=1...")
    print("   This will trigger the .map() function.")
    print("-" * 20)
    
    try:
        dataset = get_training_dataset(
            train_files=[dummy_file_path],
            tokenizer=tokenizer,
            max_seq_length=128,
            num_workers=1 # This is the critical parameter we are testing
        )
        print("-" * 20)
        print("\n4. Verification Result:")
        print("   ✅ SUCCESS! The function completed without deadlocking.")
        print(f"   Processed dataset contains {len(dataset)} samples.")
    except Exception as e:
        print("-" * 20)
        print("\n4. Verification Result:")
        print(f"   ❌ FAILED! An error occurred during the process: {e}")
    finally:
        # 5. Clean up the dummy file
        print("\n5. Cleaning up dummy data file.")
        os.remove(dummy_file_path)
        print("="*50)
        print("✅ Verification finished.")
        print("="*50)

if __name__ == "__main__":
    run_verification()