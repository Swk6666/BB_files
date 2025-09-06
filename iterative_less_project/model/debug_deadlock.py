# /root/iterative_less_project/debug_deadlock.py
import argparse
import os
import sys
import time
import signal
import threading
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader

# Add project root to path to find our modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from externals.less.data_selection.get_training_dataset import get_training_dataset

# --- Configuration (should match your main experiment) ---
MODEL_PATH = "/root/autodl-tmp/results/IterativeLess_Llama-7B_Final_Run_Robust/warmup_model"
BASE_MODEL_PATH = "/root/autodl-tmp/modelscope_cache/modelscope/Llama-2-7b-ms"
TRAIN_FILE = "/root/iterative_less_project/data/candidate_pool.jsonl"
TIMEOUT_SECONDS = 90  # If any stage takes longer than this, it's a deadlock

# --- Timeout and Monitoring Tools ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException(f"Operation timed out after {TIMEOUT_SECONDS} seconds!")

def monitor_processes(stop_event):
    """A thread that prints the current process and any children every 2 seconds."""
    current_process = psutil.Process()
    pid = current_process.pid
    print(f"[MONITOR] Started. Main PID: {pid}. Watching for children...")
    while not stop_event.is_set():
        try:
            children = current_process.children(recursive=True)
            if children:
                child_pids = [p.pid for p in children]
                print(f"[MONITOR] Main PID {pid} has {len(children)} child process(es): {child_pids}")
            else:
                print(f"[MONITOR] Main PID {pid} has no children.")
        except psutil.NoSuchProcess:
            break
        time.sleep(2)
    print("[MONITOR] Stopped.")

# --- Main Diagnostic Function ---
def run_diagnostic():
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_processes, args=(stop_monitoring,))
    
    try:
        monitor_thread.start()
        signal.signal(signal.SIGALRM, timeout_handler)

        # STAGE 1: Load Tokenizer (CPU-bound, should be fast)
        print("\n--- STAGE 1: Loading Tokenizer (CPU) ---")
        signal.alarm(TIMEOUT_SECONDS)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        signal.alarm(0)
        print("✅ STAGE 1: Tokenizer loaded successfully.\n")

        # STAGE 2: Process Dataset using .map() (CPU-bound, potential fork point)
        print("--- STAGE 2: Processing Dataset with .map() (num_proc=1) ---")
        signal.alarm(TIMEOUT_SECONDS)
        processed_dataset_object = get_training_dataset(TRAIN_FILE, tokenizer, 1024, num_workers=1)
        signal.alarm(0)
        print(f"✅ STAGE 2: Dataset processed successfully. Contains {len(processed_dataset_object)} samples.\n")

        # STAGE 3: Eagerly load into a Python list (CPU/RAM-bound)
        print("--- STAGE 3: Eagerly loading all samples into a Python list ---")
        signal.alarm(TIMEOUT_SECONDS)
        list_of_samples = [processed_dataset_object[i] for i in range(len(processed_dataset_object))]
        signal.alarm(0)
        print(f"✅ STAGE 3: All {len(list_of_samples)} samples loaded into RAM.\n")
        
        # STAGE 4: Create DataLoader (CPU-bound, should be instant)
        print("--- STAGE 4: Creating PyTorch DataLoader ---")
        signal.alarm(TIMEOUT_SECONDS)
        dataloader = DataLoader(list_of_samples, batch_size=1, shuffle=False)
        signal.alarm(0)
        print("✅ STAGE 4: DataLoader created successfully.\n")

        # --- DANGER ZONE: CUDA Initialization ---
        print("="*60)
        print("      >>> DANGER ZONE <<<")
        print("The following steps will initialize CUDA and interact with the GPU.")
        print("The deadlock, if it exists, is likely to occur from here on.")
        print("="*60)

        # STAGE 5: Load Model to GPU (This initializes CUDA)
        print("\n--- STAGE 5: Loading Model to GPU (CUDA Initialization) ---")
        signal.alarm(TIMEOUT_SECONDS * 2) # Model loading can be slower
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
            device_map={'': 'cuda'}, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, MODEL_PATH, is_trainable=True)
        signal.alarm(0)
        print("✅ STAGE 5: Model loaded to GPU successfully.\n")

        # STAGE 6: The Moment of Truth - Iterating the DataLoader
        print("--- STAGE 6: Attempting to get the FIRST batch from DataLoader ---")
        print("   If the script hangs here and times out, the deadlock is in the DataLoader iteration.")
        
        signal.alarm(TIMEOUT_SECONDS)
        
        # We will iterate just one step
        first_batch = None
        for i, batch in enumerate(dataloader):
            print(f"   ... Successfully retrieved batch {i}!")
            first_batch = batch
            if i == 0:
                break
        
        signal.alarm(0)
        
        if first_batch is None:
            raise RuntimeError("Failed to retrieve any batches from the DataLoader, but no timeout occurred.")
            
        print("✅ STAGE 6: Successfully retrieved the first batch from DataLoader!\n")

        print("="*60)
        print("🎉 DIAGNOSTIC PASSED: All stages completed without deadlock.")
        print("   This indicates the issue might not be a simple deadlock and could be related")
        print("   to resource competition in the main experiment script.")
        print("="*60)

    except TimeoutException as e:
        print("\n" + "="*60)
        print("🔥🔥🔥 DEADLOCK DETECTED! 🔥🔥🔥")
        print(f"ERROR: {e}")
        print("The script was stuck at the stage printed just before this message.")
        print("This is the exact location of your problem.")
        print("="*60)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        stop_monitoring.set()
        monitor_thread.join()

if __name__ == "__main__":
    run_diagnostic()