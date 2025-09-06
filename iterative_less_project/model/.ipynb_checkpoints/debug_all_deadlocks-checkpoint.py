# /root/iterative_less_project/debug_all_deadlocks.py
import os
import sys
import time
import signal
import psutil
import torch
import subprocess
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader

# --- Pre-script check to ensure this is run as the main program ---
if __name__ != "__main__":
    raise RuntimeError("This script is designed to be run directly, not imported.")

# Add project root to path to find our modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from externals.less.data_selection.get_training_dataset import get_training_dataset
from iterative_less.config import BASE_MODEL_NAME, CANDIDATE_POOL_FILENAME

# --- Configuration ---
MODEL_PATH = "/root/autodl-tmp/results/IterativeLess_Llama-7B_Final_Run_Robust/warmup_model"
TIMEOUT_SECONDS = 30 # Use a shorter timeout for faster failure detection

# --- Timeout Tool ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException(f"Operation timed out after {TIMEOUT_SECONDS} seconds!")

# --- Helper function to run a code block with a timeout ---
def run_with_timeout(stage_name, func):
    print(f"\n--- RUNNING: {stage_name} ---")
    print(f"    (Timeout set to {TIMEOUT_SECONDS} seconds)")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        func()
        signal.alarm(0)
        print(f"✅ PASS: Stage '{stage_name}' completed successfully without timeout.")
        return True
    except TimeoutException:
        print(f"🔴 FAIL: Stage '{stage_name}' HUNG and timed out.")
        return False
    except Exception as e:
        print(f"❌ ERROR: Stage '{stage_name}' failed with an unexpected error: {e}")
        return False

# --- Main Diagnostic Function ---
def run_all_diagnostics():
    print("="*80)
    print("🔬 STARTING COMPREHENSIVE DEADLOCK DIAGNOSTIC 🔬")
    print("="*80)

    # --- CHECK 1: Multiprocessing Start Method ---
    print("\n--- CHECK 1: Multiprocessing Start Method ---")
    start_method = mp.get_start_method(allow_none=True)
    print(f"INFO: Default process start method is: '{start_method}' (None implies 'fork' on Linux)")
    if start_method != 'spawn':
        print("🔴 DIAGNOSIS: System default is NOT 'spawn'. This is a high-risk factor for CUDA deadlocks.")
    else:
        print("✅ DIAGNOSIS: System default is 'spawn'. This is the safest setting.")

    # --- CHECK 2: `datasets.map()` Deadlock Test ---
    def datasets_map_test():
        print("    - Step 2.1: Loading tokenizer (CPU)...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        print("    - Step 2.2: Initializing CUDA by putting a dummy tensor on GPU...")
        _ = torch.randn(1, device="cuda")
        print("    - Step 2.3: Calling get_training_dataset (which uses .map)...")
        _ = get_training_dataset(CANDIDATE_POOL_FILENAME, tokenizer, 1024, num_workers=1, sample_percentage=0.01) # Use 1% of data for speed
        print("    - Step 2.4: .map() call finished.")
    
    is_map_ok = run_with_timeout("Check 2: `datasets.map()` Deadlock", datasets_map_test)
    torch.cuda.empty_cache() # Clean up GPU for the next test

    # --- CHECK 3: `DataLoader` Iteration Deadlock Test ---
    def dataloader_iteration_test():
        print("    - Step 3.1: Doing all CPU/RAM work first...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        processed_obj = get_training_dataset(CANDIDATE_POOL_FILENAME, tokenizer, 1024, num_workers=1, sample_percentage=0.01)
        list_of_samples = [processed_obj[i] for i in range(len(processed_obj))]
        print("    - Step 3.2: Loading model to GPU (Initializing CUDA)...")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map={'': 'cuda'})
        _ = PeftModel.from_pretrained(model, MODEL_PATH)
        print("    - Step 3.3: Creating DataLoader...")
        dataloader = DataLoader(list_of_samples, batch_size=1)
        print("    - Step 3.4: Attempting to get the FIRST batch from DataLoader...")
        _ = next(iter(dataloader))
        print("    - Step 3.5: First batch retrieved successfully.")

    is_dataloader_ok = run_with_timeout("Check 3: `DataLoader` Iteration Deadlock", dataloader_iteration_test)
    torch.cuda.empty_cache() # Clean up GPU

    # --- CHECK 4: Pipe Buffer Deadlock Test ---
    def pipe_buffer_test():
        # Create a dummy script that prints a lot of lines
        dummy_script_path = "dummy_child_for_pipe_test.py"
        with open(dummy_script_path, "w") as f:
            f.write("import time\n")
            f.write("for i in range(20000):\n")
            f.write("    print(f'This is line {i} of output from the child process.')\n")
            f.write("time.sleep(1)\n") # Sleep to ensure it's running when parent checks
        
        print("    - Step 4.1: Simulating parent waiting for child to finish (the WRONG way)...")
        process = subprocess.Popen(
            f"python {dummy_script_path}", shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # This will hang if the pipe buffer fills up before the process ends
        process.communicate()
        os.remove(dummy_script_path)
        print("    - Step 4.2: Parent successfully communicated.")

    is_pipe_ok = run_with_timeout("Check 4: Pipe Buffer Deadlock", pipe_buffer_test)
    
    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print("🔬 FINAL DIAGNOSTIC REPORT 🔬")
    print("="*80)
    print(f"  - Multiprocessing Start Method Risk: {'HIGH' if start_method != 'spawn' else 'LOW'}")
    print(f"  - `datasets.map()` Deadlock Test:      {'PASSED' if is_map_ok else 'FAILED'}")
    print(f"  - `DataLoader` Iteration Test:       {'PASSED' if is_dataloader_ok else 'FAILED'}")
    print(f"  - Pipe Buffer Deadlock Test:         {'PASSED' if is_pipe_ok else 'FAILED'}")
    print("="*80)

    # Conclusive Diagnosis
    print("\n--- CONCLUSIVE DIAGNOSIS ---")
    if not is_pipe_ok:
        print("🔴 The root cause is a PIPE BUFFER DEADLOCK.")
        print("   The main script waits for the subprocess to finish, but the subprocess")
        print("   freezes because it has generated too much log/progress output, filling the")
        print("   communication pipe. The solution is to read the output in real-time.")
    elif not is_dataloader_ok:
        print("🔴 The root cause is a DATALOADER ITERATION DEADLOCK.")
        print("   Even when using a simple list, iterating the DataLoader after CUDA is")
        print("   initialized is causing a hang. The solution is to remove the DataLoader")
        print("   entirely and iterate over the list manually.")
    elif not is_map_ok:
         print("🔴 The root cause is a DATASETS.MAP() DEADLOCK.")
         print("   Even with num_proc=1, the .map() function is not safe to run after")
         print("   CUDA has been initialized. The solution is to always process data")
         print("   before loading any models to the GPU.")
    elif start_method != 'spawn':
        print("🔴 The root cause is the PROCESS START METHOD.")
        print("   Although the individual tests passed in isolation, the combination of")
        print("   a 'fork' start method and the complexity of the main script is causing the")
        print("   deadlock. The solution is to force the start method to 'spawn'.")
    else:
        print("✅ All known deadlock conditions have passed. This is highly unusual.")
        print("   The problem may lie in a more complex interaction not covered here.")

    print("="*80)

if __name__ == "__main__":
    run_all_diagnostics()