import argparse
import pandas as pd
import concurrent.futures
import time
import logging
from applications.LLM.code.data_process.Multiple_interface_joint_calls.call_different_api.use_aili_qwen_api_single import qwen_main
from applications.LLM.code.data_process.Multiple_interface_joint_calls.call_different_api.use_doubao_api_single import doubao_main
from applications.LLM.code.data_process.Multiple_interface_joint_calls.call_different_api.use_gpt4o_api_single import main_gpt4o

# 配置日志
logging.basicConfig(filename='log/main.log', level=logging.INFO, encoding="utf-8", filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 定义处理函数，调用各个API并统计时间
def process_and_time(api_func, df, api_name, column_name):
    """
    调用指定的API函数并统计其运行时间。

    参数:
    api_func (function): 要调用的API函数
    df (pd.DataFrame): 输入的DataFrame
    api_name (str): API名称，用于打印和日志记录
    column_name (str): 要在DataFrame中新增的列名

    返回:
    pd.DataFrame: 新增列后的DataFrame
    float: API调用的运行时间
    """
    try:
        logging.info(f"Starting {api_name}")
        start_time = time.time()  # 记录开始时间
        result_df = api_func(df.copy())  # 调用API函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        logging.info(f"{api_name} completed in {elapsed_time:.2f} seconds")  # 打印运行时间
        df[column_name] = result_df[column_name]  # 将结果列添加到原始DataFrame
        return df, elapsed_time  # 返回新增列后的DataFrame和运行时间
    except Exception as e:
        error_message = f"Error occurred in {api_name}: {e}"
        logging.error(error_message)  # 记录错误信息到日志文件
        df[column_name] = "error"  # 在DataFrame中标记错误
        return df, 0  # 返回标记错误的DataFrame和0时间表示失败


# 主函数
def main(input_file, output_file, models):
    """
    主函数，读取输入文件，调用指定的API处理数据，并将结果保存到输出文件。

    参数:
    input_file (str): 输入的Excel文件路径
    output_file (str): 输出的Excel文件路径
    models (list of str): 要运行的API模型列表，例如 ['qwen', 'doubao', 'gpt4o']
    """
    logging.info("Reading input file")
    # 读取Excel文件到DataFrame
    df = pd.read_excel(input_file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建future任根据models参数决定调用哪些API
        futures = {}
        # todo:将这部分写到command命令行中，只用命令行来实现调度。
        if "qwen" in models:
            futures[executor.submit(process_and_time, qwen_main, df.copy(), "Qwen API",
                                    "qwen_long_output")] = "qwen_long_output"
        if "doubao" in models:
            futures[executor.submit(process_and_time, doubao_main, df.copy(), "Doubao API",
                                    "doubao_output_new")] = "doubao_output_new"
        if "gpt4o" in models:
            futures[
                executor.submit(process_and_time, main_gpt4o, df.copy(), "GPT-4O API", "gpt4o_output")] = "gpt4o_output"

        for future in concurrent.futures.as_completed(futures):
            column_name = futures[future]
            try:
                result_df, elapsed_time = future.result()  # 获取API调用结果和运行时间
                df[column_name] = result_df[column_name]  # 更新DataFrame的对应列
            except Exception as e:
                error_message = f"Error occurred while processing {column_name}: {e}"
                logging.error(error_message)  # 记录错误信息到日志文件
                df[column_name] = "error"  # 在DataFrame中标记错误

    # 保存结果到Excel文件
    logging.info("Saving output file")
    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="Process some Excel files with different APIs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Excel file")  # 输入文件路径
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Excel file")  # 输出文件路径
    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        required=True,
        help="List of models to run (e.g., qwen, doubao, gpt4o)"  # 要运行的API模型列表
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.input_file, args.output_file, args.models)
