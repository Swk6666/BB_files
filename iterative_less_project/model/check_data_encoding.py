# /root/iterative_less_project/check_data_encoding.py
import os
import argparse
import logging
from typing import Tuple, Optional

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def check_file_encoding(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    尝试以严格的UTF-8编码读取文件，检查是否存在编码问题。

    Args:
        file_path: 文件的完整路径。

    Returns:
        一个元组 (is_ok, error_message)。
        如果文件是合法的UTF-8，返回 (True, None)。
        如果文件存在编码问题，返回 (False, "具体的错误信息")。
    """
    try:
        with open(file_path, 'rb') as f:
            # 读取所有字节
            content_bytes = f.read()
            # 尝试用UTF-8解码，如果失败会触发UnicodeDecodeError
            content_bytes.decode('utf-8')
        return True, None
    except UnicodeDecodeError as e:
        # 捕获到解码错误，说明文件不是纯UTF-8
        return False, str(e)
    except Exception as e:
        # 捕获其他可能的读取错误
        return False, f"An unexpected error occurred: {e}"

def main():
    """
    主函数，遍历MMLU数据目录并检查所有.csv文件的编码。
    """
    parser = argparse.ArgumentParser(
        description="检查MMLU数据集中所有.csv文件的UTF-8编码是否正确。"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help="MMLU数据集所在的根目录路径。"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logger.error(f"错误：指定的目录不存在 -> {args.data_dir}")
        return

    logger.info(f"--- 开始扫描目录 '{args.data_dir}' 中的.csv文件 ---")

    problematic_files = []
    ok_files_count = 0
    total_files_checked = 0

    # 使用os.walk遍历所有子目录
    for root, _, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".csv"):
                total_files_checked += 1
                file_path = os.path.join(root, filename)
                is_ok, error_message = check_file_encoding(file_path)

                if is_ok:
                    ok_files_count += 1
                else:
                    problematic_files.append((file_path, error_message))

    print("\n" + "="*80)
    logger.info("--- 扫描完成，生成报告 ---")
    print("="*80)

    if problematic_files:
        logger.warning(f"检测到 {len(problematic_files)} 个存在编码问题的文件！")
        print("-" * 80)
        for path, error in problematic_files:
            print(f"❌ 文件: {path}")
            print(f"   错误详情: {error}\n")
        print("-" * 80)
    else:
        logger.info("🎉 恭喜！所有扫描到的.csv文件都是合法的UTF-8编码。")

    print("\n--- 总结 ---")
    print(f"总共检查文件数: {total_files_checked}")
    print(f"  - ✅ 编码正确: {ok_files_count}")
    print(f"  - ❌ 存在问题: {len(problematic_files)}")
    print("="*80)

    if problematic_files:
        logger.info("建议：您可以手动打开这些有问题的文件，查找并删除导致错误的特殊字符，或者在 `run_eval.py` 的 pd.read_csv 中添加 encoding_errors='ignore' 参数来跳过这些错误。")


if __name__ == "__main__":
    main()