# /root/iterative_less_project/child_for_debug.py
import torch
import argparse
import time
import sys

def run_cuda_and_io():
    print("Child: Starting mode 'cuda_and_io'. Will print and do CUDA work.")
    print(f"Child: CUDA is available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Child: Cannot run this test without a CUDA device.", file=sys.stderr)
        sys.exit(1)
        
    for i in range(500): # 循环次数不需要太多，足以触发死锁
        # 模拟 CUDA 计算
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        
        # 模拟 tqdm 的高频打印
        print(f"Child: Processing step {i+1}/500")
        
        # 确保输出被立即发送，而不是被Python的缓冲区卡住
        sys.stdout.flush() 
    print("Child: Finished 'cuda_and_io' successfully.")

def run_cuda_only():
    print("Child: Starting mode 'cuda_only'. Will only do CUDA work, no printing in loop.")
    for i in range(500):
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        # No print statement here
    print("Child: Finished 'cuda_only' successfully.")

def run_io_only():
    print("Child: Starting mode 'io_only'. Will only print, no CUDA work in loop.")
    for i in range(500):
        print(f"Child: Processing step {i+1}/500")
        sys.stdout.flush()
    print("Child: Finished 'io_only' successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["cuda_and_io", "cuda_only", "io_only"])
    args = parser.parse_args()

    if args.mode == "cuda_and_io":
        run_cuda_and_io()
    elif args.mode == "cuda_only":
        run_cuda_only()
    elif args.mode == "io_only":
        run_io_only()