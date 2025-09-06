# /root/iterative_less_project/iterative_less/progress_logger.py
import json
import os
import time
import atexit

class ProgressLogger:
    """
    一个简单的基于文件的进度记录器，用于跨进程通信。
    子进程写入状态，主进程读取状态。
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.pid = os.getpid()
        # 确保在进程退出时自动清理
        atexit.register(self.cleanup)
        # 初始化日志文件
        self.log("Initializing...")

    def log(self, message: str, details: str = ""):
        """
        将当前状态写入日志文件。
        """
        try:
            status = {
                "pid": self.pid,
                "timestamp": time.time(),
                "message": message,
                "details": details
            }
            # 原子写入：先写入临时文件，再重命名
            temp_path = self.log_file_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(status, f)
            os.rename(temp_path, self.log_file_path)
        except Exception:
            # 在退出或清理阶段可能会失败，静默处理
            pass

    def cleanup(self):
        """
        清理日志文件。
        """
        try:
            if os.path.exists(self.log_file_path):
                os.remove(self.log_file_path)
            if os.path.exists(self.log_file_path + ".tmp"):
                os.remove(self.log_file_path + ".tmp")
        except (FileNotFoundError, PermissionError):
            pass