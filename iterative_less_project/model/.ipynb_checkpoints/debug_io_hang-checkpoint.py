# /root/iterative_less_project/debug_io_hang.py
import subprocess
import os
import time
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置区 ---
# 请确保这些路径与您实际环境中 `main_less.py` 运行时使用的路径一致。
WARMUP_MODEL_PATH = "/root/autodl-tmp/results/IterativeLess_Llama-7B_Final_Run_Robust/warmup_model"
BASE_MODEL_PATH = "/root/autodl-tmp/modelscope_cache/modelscope/Llama-2-7b-ms"
CANDIDATE_POOL_PATH = "/root/iterative_less_project/data/candidate_pool.jsonl"
DEBUG_OUTPUT_PATH = "/root/iterative_less_project/debug_grad_output"
GRAD_PROJ_DIM = 8192
GRAD_TYPE = "adam"

# --- 【核心修正】为子进程创建一个临时的progress_log文件 ---
PROGRESS_LOG_PATH_FOR_SUBPROCESS = "/tmp/debug_subprocess_progress.json"

# 设置一个超时时间（秒），如果超过这个时间子进程还没结束，我们就认为它挂起了。
# 梯度计算需要时间，我们将超时延长到 10 分钟 (600秒) 以确保不是因为计算慢而被误判
TIMEOUT_SECONDS = 600

def main():
    """
    修正版的独立Debug脚本。
    它现在会正确地为子进程提供所有必需的参数，使其能够真正开始执行梯度计算任务，
    从而有效地测试I/O死锁假说。
    """
    
    # 在开始前清理上一次运行可能留下的文件
    if os.path.exists(DEBUG_OUTPUT_PATH):
        shutil.rmtree(DEBUG_OUTPUT_PATH)
    if os.path.exists(PROGRESS_LOG_PATH_FOR_SUBPROCESS):
        os.remove(PROGRESS_LOG_PATH_FOR_SUBPROCESS)

    # 1. 构造一个完全正确的命令行
    command_parts = [
        "python", "-m", "externals.less.data_selection.get_info",
        "--model_path", WARMUP_MODEL_PATH,
        "--base_model_path", BASE_MODEL_PATH,
        "--train_file", CANDIDATE_POOL_PATH,
        "--output_path", DEBUG_OUTPUT_PATH,
        "--info_type", "grads",
        "--gradient_type", GRAD_TYPE,
        "--gradient_projection_dimension", str(GRAD_PROJ_DIM),
        # --- 【核心修正】添加缺失的必需参数 ---
        "--progress_log_file", PROGRESS_LOG_PATH_FOR_SUBPROCESS,
    ]
    command = " ".join(command_parts)
    
    logger.info("--- 开始修正版的独立Debug测试 ---")
    logger.info("本脚本将尝试复现I/O挂起问题。")
    logger.info(f"将在 {TIMEOUT_SECONDS} 秒后超时。")
    logger.info(f"将要执行的命令:\n{command}\n")

    # 2. 启动子进程
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    start_time = time.time()
    
    try:
        # 3. 模拟有问题的I/O处理方式，只读取stdout
        while process.poll() is None:
            if time.time() - start_time > TIMEOUT_SECONDS:
                logger.error("!!! HANG DETECTED !!!")
                logger.error(f"子进程运行超过 {TIMEOUT_SECONDS} 秒仍未结束，确认为挂起/死锁。")
                logger.error("诊断结论：I/O死锁假说得到证实。子进程的stderr管道很可能被tqdm进度条写满并阻塞。")
                return 

            try:
                line = process.stdout.readline()
                if line:
                    # 打印一些stdout输出来确认子进程至少在工作
                    logger.info(f"[SUBPROCESS STDOUT] {line.strip()}")
            except Exception:
                pass

            time.sleep(1) # 轮询间隔

        # 4. 如果程序能正常走到这里，说明子进程已经自己结束了
        logger.info("--- 子进程已在超时前正常结束 ---")
        logger.info("诊断结论：在本次测试中未检测到I/O死锁。")
        
        stdout, stderr = process.communicate()
        if stdout:
            logger.info("--- 剩余 STDOUT --- \n" + stdout)
        if stderr:
            logger.info("--- 剩余 STDERR --- \n" + stderr)
        
        if process.returncode != 0:
            logger.warning(f"子进程虽然正常结束，但返回了错误码: {process.returncode}")

    finally:
        # 确保无论如何都清理子进程
        if process.poll() is None:
            logger.warning("正在强制终止挂起的子进程...")
            process.kill()
            logger.warning("子进程已终止。")
        
        # 清理生成的临时文件夹和文件
        if os.path.exists(DEBUG_OUTPUT_PATH):
            shutil.rmtree(DEBUG_OUTPUT_PATH)
        if os.path.exists(PROGRESS_LOG_PATH_FOR_SUBPROCESS):
            os.remove(PROGRESS_LOG_PATH_FOR_SUBPROCESS)
        logger.info("已清理所有临时文件和目录。")

if __name__ == "__main__":
    main()