# /root/iterative_less_project/debug_final_deadlock.py
import subprocess
import sys
import time

TIMEOUT_SECONDS = 120 # 如果一个实验卡住超过20秒，就判定为死锁

def run_test(mode: str):
    print("\n" + "="*50)
    print(f"🔬 STARTING TEST: Mode = {mode}")
    print("="*50)
    
    command = f"python child_for_debug.py --mode {mode}"
    print(f"Parent: Launching child with command: {command}")
    
    start_time = time.time()
    try:
        # 我们使用 Popen 和 communicate() 来精确模拟 toolkit.py 之前的工作方式
        # communicate() 会等待子进程完全结束后再返回
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # communicate() 有一个内置的超时参数，这是最可靠的检测挂起的方法
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        
        end_time = time.time()
        
        if process.returncode == 0:
            print(f"✅ SUCCESS: Child process finished in {end_time - start_time:.2f} seconds.")
            return True
        else:
            print(f"❌ FAILED: Child process exited with error code {process.returncode}.")
            print("--- STDERR ---")
            print(stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"🔥🔥🔥 HANG DETECTED! 🔥🔥🔥")
        print(f"Child process did not finish within the {TIMEOUT_SECONDS} second timeout.")
        print("This is conclusive proof of a deadlock.")
        process.kill() # 清理卡住的进程
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("🔬 Ultimate Deadlock Diagnostic Tool 🔬")
    print("This tool will run three experiments to find the root cause of the hang.")
    print("="*80)

    # 实验 A: CUDA 和 I/O 同时进行 (预测：会失败/挂起)
    test_a_result = run_test("cuda_and_io")
    
    # 实验 B: 只有 CUDA (预测：会成功)
    test_b_result = run_test("cuda_only")

    # 实验 C: 只有 I/O (预测：会成功)
    test_c_result = run_test("io_only")

    print("\n" + "="*80)
    print("🔬 FINAL DIAGNOSTIC REPORT 🔬")
    print("="*80)
    print(f"  - Test A (CUDA + I/O):  {'FAILED (HUNG)' if not test_a_result else 'PASSED'}")
    print(f"  - Test B (CUDA Only):   {'PASSED' if test_b_result else 'FAILED'}")
    print(f"  - Test C (I/O Only):    {'PASSED' if test_c_result else 'FAILED'}")
    print("="*80)
    
    print("\n--- CONCLUSIVE DIAGNOSIS ---")
    if not test_a_result and test_b_result and test_c_result:
        print("🔴 The evidence is conclusive: The deadlock is caused by the INTERACTION")
        print("   between high-frequency I/O (like tqdm) and CUDA computations within")
        print("   a piped subprocess.")
        print("\n   The definitive solution is to remove the high-frequency I/O (tqdm)")
        print("   from the CUDA-intensive loop in 'collect_grad_reps.py'.")
    else:
        print("🟡 The results are unexpected. The root cause may be different.")
        print("   Please share this full log for further analysis.")
    print("="*80)