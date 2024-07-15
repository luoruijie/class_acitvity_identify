import pandas as pd
import random
from http import HTTPStatus
import dashscope
import logging
from data_process_public import prompt

# 配置日志
logging.basicConfig(filename='../log/qwen.log', level=logging.INFO, encoding='utf-8', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

dashscope.api_key = "sk-f089718a48534c4c84a0cfbc35e9fd1a"


# 定义处理函数
def process_text_list(df):
    for i in range(len(df)):
        try:
            user_input = df.loc[i, 'text']
            messages = [{'role': 'system', 'content': prompt},
                        {'role': 'user', 'content': user_input}]

            response1 = dashscope.Generation.call(
                model="qwen-long",
                messages=messages,
                seed=random.randint(1, 10000),  # 设置随机数种子
                result_format='message'
            )
            print('response1:', response1)
            if response1.status_code == HTTPStatus.OK:
                content = response1["output"]['choices'][0]['message']['content']
                print("cotent", content)
                df.loc[i, 'predict'] = content
            else:
                logging.error(f"Error response status code: {response1.status_code} for input: {user_input}")
                df.loc[i, 'predict'] = "Error"

        except Exception as e:
            logging.error(f"Error processing text at index {i}: {e}")
            df.loc[i, 'qwen_long_output'] = "Error"

    return df


# 主函数
def qwen_main(input_df):
    try:
        result_df = process_text_list(input_df)
        return result_df
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        return input_df


# 测试主函数
if __name__ == "__main__":
    try:
        df = pd.read_excel("../../data/test_llm/高希娜.xlsx")
        df = df[0:2]
        result_df = qwen_main(df)
        result_df.to_excel("output.xlsx", index=False)
    except Exception as e:
        logging.error(f"Error reading or writing Excel file: {e}")
