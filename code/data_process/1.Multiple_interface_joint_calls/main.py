import argparse
import pandas as pd
import concurrent.futures
import time
import logging
import importlib
import json
import os

# 配置日志
logging.basicConfig(filename='log/main.log', level=logging.INFO, encoding="utf-8", filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

    功能说明:
    - 调用指定的API函数并统计其运行时间。
    - 处理输入的DataFrame，并将API调用的结果添加到新的列中。
    - 记录API调用的开始和结束时间，以及在日志中记录任何错误。
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

def load_api_function(full_function_name):
    """
    动态加载指定的API函数。

    参数:
    full_function_name (str): 模块名称和函数名称字符串

    返回:
    function: 加载的API函数

    功能说明:
    - 动态加载指定的API函数。
    - 从模块名称和函数名称字符串中解析出模块和函数，并返回对应的函数对象。
    """
    module_name, function_name = full_function_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def generate_output_filename(input_file, model_name, output_dir):
    """
    生成输出文件名。

    参数:
    input_file (str): 输入的Excel文件路径
    model_name (str): 模型名称
    output_dir (str): 输出目录路径

    返回:
    str: 生成的输出文件路径

    功能说明:
    - 根据输入文件名称和模型名称生成输出文件名。
    - 输出文件名格式为 <模型名称>_<输入文件名>_output.xlsx，并将其保存在指定的输出目录中。
    """
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{model_name}_{input_filename}_output.xlsx"
    return os.path.join(output_dir, output_filename)

def main(input_file, output_dir, model_numbers, config_file):
    """
    主函数，读取输入文件，调用指定的API处理数据，并将结果保存到对应的输出文件。

    参数:
    input_file (str): 输入的Excel文件路径
    output_dir (str): 输出目录路径
    model_numbers (list of str): 要运行的API模型编号列表，例如 ['1', '2', '3']
    config_file (str): 配置文件路径

    功能说明:
    - 读取输入的Excel文件。
    - 读取配置文件，获取模型编号到模型名称的映射和API配置信息。
    - 使用多线程并发调用各个API处理数据。
    - 根据输入文件名称和模型名称生成输出文件名，并将处理结果保存到对应的输出文件中。
    """
    logging.info("Reading input file")
    # 读取Excel文件到DataFrame
    df = pd.read_excel(input_file)

    # 读取配置文件
    with open(config_file, 'r') as file:
        config = json.load(file)

    # 获取模型编号到模型名称的映射
    model_mapping = config["model_mapping"]
    model_config = config["models"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建future任根据models参数决定调用哪些API
        futures = {}
        for number in model_numbers:
            model_name = model_mapping.get(number)
            if model_name and model_name in model_config:
                api_func = load_api_function(model_config[model_name]['api_func'])
                column_name = model_config[model_name]['column_name']
                output_file = generate_output_filename(input_file, model_name, output_dir)
                futures[executor.submit(process_and_time, api_func, df.copy(), f"{model_name} API",
                                        column_name)] = (column_name, output_file)

        for future in concurrent.futures.as_completed(futures):
            column_name, output_file = futures[future]
            try:
                result_df, elapsed_time = future.result()  # 获取API调用结果和运行时间
                result_df.to_excel(output_file, index=False)  # 保存结果到对应的Excel文件
                logging.info(f"Saved output to {output_file}")
            except Exception as e:
                error_message = f"Error occurred while processing {column_name}: {e}"
                logging.error(error_message)  # 记录错误信息到日志文件

if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="Process some Excel files with different APIs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Excel file")  # 输入文件路径
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output Excel files"  # 输出目录路径
    )
    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        required=True,
        help="List of model numbers to run (e.g., 1, 2, 3)"  # 要运行的API模型编号列表
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file"  # 配置文件路径
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.input_file, args.output_dir, args.models, args.config_file)
