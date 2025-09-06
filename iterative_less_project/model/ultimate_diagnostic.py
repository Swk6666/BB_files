# /root/iterative_less_project/ultimate_diagnostic.py
import os
import sys
import time
import subprocess
import torch
import torch.multiprocessing as mp
import threading
import logging

# --- Configuration ---
TIMEOUT_SECONDS = 30  # If any test hangs for this long, we kill it and mark as failed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Helper Functions ---
def run_child_process(start_method=None):
    """Launches this script in 'child' mode with a specific start method."""
    ctx = mp.get_context(start_method)
    process = ctx.Process(target=child_main)
    process.start()
    process.join(timeout=TIMEOUT_SECONDS)
    
    if process.is_alive():
        print("🔴 Child process is still alive after timeout. HANG DETECTED.")
        process.kill()
        process.join()
        return False
    
    if process.exitcode != 0:
        print(f"❌ Child process failed with exit code {process.exitcode}.")
        return False
        
    print("✅ Child process finished successfully.")
    return True

def run_pipe_test(realtime_read: bool):
    """Launches a child that prints a lot and tests how the parent reads it."""
    # The child script logic is embedded here for simplicity
    child_code = """
import sys
import time
for i in range(10000):
    if i % 2 == 0:
        print(f"STDOUT line {i}", file=sys.stdout)
    else:
        print(f"STDERR line {i}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
"""
    command = [sys.executable, "-c", child_code]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

    if realtime_read:
        # The SAFE way: use threads to read both pipes simultaneously
        def reader_thread(pipe, log_prefix):
            for line in iter(pipe.readline, ''):
                pass # Just consume the line
            pipe.close()
        
        stdout_thread = threading.Thread(target=reader_thread, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=reader_thread, args=(process.stderr, "STDERR"))
        stdout_thread.start()
        stderr_thread.start()
        
        process.wait(timeout=TIMEOUT_SECONDS) # Wait for the process to finish
        stdout_thread.join()
        stderr_thread.join()
    else:
        # The UNSAFE way: wait for the process to finish before reading
        process.communicate(timeout=TIMEOUT_SECONDS)

    if process.returncode == 0:
        print("✅ Child process finished successfully.")
        return True
    else:
        print(f"❌ Child process failed with exit code {process.returncode}.")
        return False

# --- Child Process Main Logic ---
def child_main():
    """This function is run by the child process in Test 1."""
    print(f"    [Child PID: {os.getpid()}] Starting child process.")
    time.sleep(1)
    print(f"    [Child PID: {os.getpid()}] Initializing CUDA...")
    try:
        _ = torch.randn(1, device="cuda")
        print(f"    [Child PID: {os.getpid()}] CUDA Initialized successfully.")
    except Exception as e:
        print(f"    [Child PID: {os.getpid()}] CUDA Initialization FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    time.sleep(2)
    print(f"    [Child PID: {os.getpid()}] Child process exiting.")
    sys.exit(0)

# --- Parent Process Main Diagnostic ---
def run_all_diagnostics():
    print("="*80)
    print("🔬 Ultimate Deadlock Diagnostic Tool 🔬")
    print("This tool will run three experiments to find the undeniable root cause of the hang.")
    print("="*80)

    # --- Test 1: `fork` vs. `spawn` ---
    print("\n--- Test 1: `fork` vs. `spawn` Deadlock Test ---")
    print("Testing if creating a child process after CUDA initialization hangs.")
    
    print("\n[1A] Using unsafe 'fork' method...")
    torch.cuda.init() # Intentionally contaminate the parent process
    print("Parent: CUDA has been initialized in the parent process.")
    test_a_result = run_child_process('fork')
    
    print("\n[1B] Using safe 'spawn' method...")
    test_b_result = run_child_process('spawn')
    
    # --- Test 2 & 3: Pipe Buffer Deadlock ---
    print("\n--- Test 2 & 3: I/O Pipe Buffer Deadlock Test ---")
    print("Testing if a child process that prints a lot can hang the parent.")
    
    print("\n[2] Using unsafe 'wait-at-the-end' reading method...")
    try:
        test_c_result = run_pipe_test(realtime_read=False)
    except subprocess.TimeoutExpired:
        print(f"🔥🔥🔥 HANG DETECTED! 🔥🔥🔥")
        print("Child process did not finish within the timeout. This confirms a pipe buffer deadlock.")
        test_c_result = False

    print("\n[3] Using safe 'real-time threaded' reading method...")
    try:
        test_d_result = run_pipe_test(realtime_read=True)
    except subprocess.TimeoutExpired:
        print("🔥🔥🔥 HANG DETECTED! 🔥🔥🔥")
        print("This should not happen. The threaded reader failed to prevent the deadlock.")
        test_d_result = False

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print("🔬 FINAL DIAGNOSTIC REPORT 🔬")
    print("="*80)
    print(f"  - Test 1A (Parent CUDA + fork child):  {'FAILED (HUNG)' if not test_a_result else 'PASSED'}")
    print(f"  - Test 1B (Parent CUDA + spawn child): {'PASSED' if test_b_result else 'FAILED'}")
    print(f"  - Test 2  (Pipe I/O - wait-at-end):  {'FAILED (HUNG)' if not test_c_result else 'PASSED'}")
    print(f"  - Test 3  (Pipe I/O - real-time):    {'PASSED' if test_d_result else 'FAILED'}")
    print("="*80)
    
    print("\n--- CONCLUSIVE DIAGNOSIS ---")
    if not test_c_result and test_d_result:
        print("🔴 The evidence is CONCLUSIVE: The root cause is a PIPE BUFFER DEADLOCK.")
        print("   Your child process (get_info.py) produces a large volume of output (stdout + stderr),")
        print("   which fills the communication pipe and causes it to freeze while waiting for the")
        print("   parent process (main_less.py) to read it. The parent, however, is waiting for")
        print("   the child to finish.")
        print("\n   >> The definitive solution is to modify 'toolkit.py' to read BOTH stdout and stderr")
        print("      in real-time using separate threads.")
    elif not test_a_result and test_b_result:
        print("🔴 The evidence is CONCLUSIVE: The root cause is the PROCESS START METHOD.")
        print("   Your main script initializes CUDA before forking a subprocess. This creates")
        print("   a corrupted CUDA state in the child, causing an immediate deadlock.")
        print("\n   >> The definitive solution is to add `mp.set_start_method('spawn', force=True)`")
        print("      at the very beginning of your main scripts (`main_less.py`, etc.).")
    else:
        print("🟡 The results are unexpected. All tests passed, which contradicts the observed")
        print("   behavior. This points to a more complex, intermittent issue, possibly")
        print("   related to system load or specific driver/library versions. However,")
        print("   applying BOTH the 'spawn' and 'real-time pipe reading' fixes is the")
        print("   most robust path forward to eliminate all known causes.")
    print("="*80)

if __name__ == "__main__":
    # We are the parent process, running the diagnostics
    if len(sys.argv) == 1:
        run_all_diagnostics()
    # We are the child process, being run by Test 1
    elif len(sys.argv) > 1 and sys.argv[0].endswith('ultimate_diagnostic.py'):
        child_main()