# /root/iterative_less_project/pre_download.py
import os
import logging
from modelscope.hub.snapshot_download import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_and_download():
    # 1. 验证环境变量
    hf_cache = os.getenv("HF_HOME")
    if not hf_cache or "autodl-tmp/cache" not in hf_cache:
        logger.error("错误：HF_HOME 环境变量未正确设置到 '/root/autodl-tmp/cache' 目录！请先运行 `source setup_env.sh`")
        return False
    logger.info(f"✅ Hugging Face 缓存目录已正确设置为: {hf_cache}")

    ms_cache = os.getenv("MODELSCOPE_CACHE")
    if not ms_cache or "autodl-tmp/cache" not in ms_cache:
        logger.error("错误：MODELSCOPE_CACHE 环境变量未正确设置到 '/root/autodl-tmp/cache' 目录！请先运行 `source setup_env.sh`")
        return False
    logger.info(f"✅ ModelScope 缓存目录已正确设置为: {ms_cache}")

    from transformers import MT5ForConditionalGeneration, T5Tokenizer

    #LLAMA_MODEL_MS_ID = "modelscope/Llama-2-7b-ms"
    MT5_MODEL_HF_ID = "google/mt5-base"

    try:
        # 1. 从 ModelScope 显式下载 Llama-2-7B 到其缓存目录
        #logger.info(f"\n--- 开始从 ModelScope 显式下载 {LLAMA_MODEL_MS_ID} ---")
        #local_llama_path = snapshot_download(
            #LLAMA_MODEL_MS_ID,
            #cache_dir=ms_cache # 明确使用环境变量指定的路径
        #)
        #logger.info(f"✅ {LLAMA_MODEL_MS_ID} 已成功下载到: {local_llama_path}")

        # 2. 从 Hugging Face 预下载 google/mt5-base 到其缓存目录
        logger.info(f"\n--- 开始从 Hugging Face 预下载 {MT5_MODEL_HF_ID} ---")
        # transformers会自动使用HF_HOME环境变量
        T5Tokenizer.from_pretrained(MT5_MODEL_HF_ID)
        MT5ForConditionalGeneration.from_pretrained(MT5_MODEL_HF_ID)
        logger.info(f"✅ {MT5_MODEL_HF_ID} 已成功下载并缓存。")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 所有模型都已成功下载到数据盘的 'cache' 目录中！")
        logger.info("下一步，请将下面的 Llama 本地路径更新到您的 config.py 文件中：")
        logger.info(f"Llama 本地路径: {local_llama_path}")
        logger.info("="*80)
        
        return True

    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    verify_and_download()