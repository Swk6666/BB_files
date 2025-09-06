import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def analyze_results(file_path: str):
    """
    读取实验结果JSON文件，生成表格和图表来分析各策略的性能变化。
    """
    logger.info(f"--- 开始分析实验结果文件: {file_path} ---")

    # 1. 加载并预处理数据
    try:
        df = pd.read_json(file_path)
    except FileNotFoundError:
        logger.error(f"错误：找不到文件 '{file_path}'。请确保文件路径正确。")
        return
    except ValueError:
        logger.error(f"错误：文件 '{file_path}' 不是一个有效的JSON文件或格式不正确。")
        return

    # 检查必要的字段是否存在
    required_fields = ['strategy', 'phase', 'accuracy_dev', 'loss_dev']
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    if missing_fields:
        logger.error(f"错误：数据文件缺少必要的字段: {', '.join(missing_fields)}")
        logger.info(f"数据文件中实际存在的字段: {', '.join(df.columns)}")
        
        # 询问用户是否要修改字段名（如果是交互式运行）
        try:
            use_custom_fields = input("是否要使用自定义字段名替代? (y/n): ").strip().lower() == 'y'
            if use_custom_fields:
                field_mapping = {}
                for field in missing_fields:
                    custom_field = input(f"请输入 '{field}' 的替代字段名: ").strip()
                    if custom_field in df.columns:
                        field_mapping[field] = custom_field
                    else:
                        logger.error(f"输入的替代字段 '{custom_field}' 不存在于数据文件中")
                        return
                
                # 重命名列
                df = df.rename(columns={v: k for k, v in field_mapping.items()})
                logger.info("已使用自定义字段名进行替代")
            else:
                return
        except EOFError:
            # 非交互式环境
            logger.error("请检查数据文件格式，确保包含所有必要的字段")
            return

    # 提取基线 (warmup_baseline 或 baseline) 作为所有策略的 Phase 0
    try:
        baseline_row = df[df['strategy'].str.contains("baseline", case=False)].iloc[0]
    except IndexError:
        logger.error("错误：找不到包含 'baseline' 的策略数据")
        return
    
    # 提取所有实际的策略名称
    strategies = sorted([s for s in df['strategy'].unique() if "baseline" not in s])
    
    if not strategies:
        logger.warning("警告：没有找到除基线外的其他策略数据")
        return
    
    # 为每个策略创建一个 Phase 0 的起始点
    baseline_entries = []
    for strategy in strategies:
        baseline_entries.append({
            'phase': 0,
            'strategy': strategy,
            'accuracy_dev': baseline_row['accuracy_dev'],
            'loss_dev': baseline_row['loss_dev']
        })
    
    baseline_df_expanded = pd.DataFrame(baseline_entries)

    # 合并基线和实际的迭代结果
    results_df = df[~df['strategy'].str.contains("baseline", case=False)]
    df_processed = pd.concat([baseline_df_expanded, results_df]).sort_values(by=['strategy', 'phase']).reset_index(drop=True)

    # 2. 生成并打印表格
    logger.info("\n" + "="*80)
    logger.info("--- 性能变化表格 ---")
    
    # 准确率表格
    accuracy_pivot = df_processed.pivot(index='strategy', columns='phase', values='accuracy_dev')
    print("\n[表1: 开发集准确率 (Dev Accuracy) 变化]")
    print(tabulate(accuracy_pivot, headers='keys', tablefmt='pipe', floatfmt=".4f"))

    # 损失表格
    loss_pivot = df_processed.pivot(index='strategy', columns='phase', values='loss_dev')
    print("\n[表2: 开发集损失 (Dev Loss) 变化]")
    print(tabulate(loss_pivot, headers='keys', tablefmt='pipe', floatfmt=".4f"))
    logger.info("="*80)

    # 3. 生成并保存图表
    logger.info("\n--- 正在生成性能变化图表 ---")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 创建一个包含两个子图的画布 (一个用于准确率，一个用于损失)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('各策略在不同迭代阶段的性能变化', fontsize=18, weight='bold')

    # 绘制准确率曲线
    sns.lineplot(data=df_processed, x='phase', y='accuracy_dev', hue='strategy', marker='o', ax=ax1, linewidth=2.5)
    ax1.set_title('开发集准确率 (越高越好)', fontsize=14)
    ax1.set_ylabel('Dev Accuracy')
    ax1.legend(title='策略')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 绘制损失曲线
    sns.lineplot(data=df_processed, x='phase', y='loss_dev', hue='strategy', marker='o', ax=ax2, linewidth=2.5)
    ax2.set_title('开发集损失 (越低越好)', fontsize=14)
    ax2.set_ylabel('Dev Loss')
    ax2.set_xlabel('迭代阶段 (Phase)', fontsize=12)
    ax2.legend(title='策略')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 调整布局，防止重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表到文件
    output_image_path = file_path.replace('.json', '_analysis.png')
    try:
        plt.savefig(output_image_path, dpi=300)
        logger.info(f"✅ 图表已成功保存到: {output_image_path}")
    except Exception as e:
        logger.error(f"错误：无法保存图表文件。原因: {e}")
    
    # 可选：如果你在本地带图形界面的环境，可以取消下面一行的注释来直接显示图表
    # plt.show()


def main():
    # 直接指定JSON文件路径，无需命令行参数
    file_path = "/root/autodl-tmp/results/IterativeLess_Llama-7B_Final_Run_Robust/experiment_results.json"
    analyze_results(file_path)

if __name__ == "__main__":
    main()
    