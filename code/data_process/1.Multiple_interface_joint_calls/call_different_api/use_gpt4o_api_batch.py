import pandas as pd  # 导入pandas库，用于数据处理
from openai import AzureOpenAI  # 导入AzureOpenAI库，用于调用OpenAI的API
import logging  # 导入logging库，用于日志记录
from data_process_public import extract_analysis, process_text, extract_json, prompt  # 从data_process_public模块导入3个函数

# 配置日志，日志文件名为'gpt4o.log'，日志级别为INFO
logging.basicConfig(filename='../log/gpt4o.log', level=logging.INFO, encoding="utf-8", filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 初始化AzureOpenAI客户端
client = AzureOpenAI(
    azure_endpoint="https://zonekey-gpt4o.openai.azure.com/",
    api_key="b01e3eb073fe43629982b30b3548c36e",
    api_version="2024-02-01"
)

unit_length = 4  # 定义每个单位的长度
step = 1  # 定义步长


# 定义处理函数
def process_text_list(df):
    total_length = len(df)  # 获取DataFrame的总长度
    unit_count = max(total_length // unit_length, 0)  # 计算单位数量，确保至少有一个单位

    messages_list = []  # 初始化消息列表

    # 循环处理每16个单位
    for i in range(unit_count + 1):  # 包括最后不足16个的部分
        messages = [{"role": "system", "content": prompt}]  # 初始化消息，添加系统消息
        for j in range(0, unit_length, step):  # 步长为1
            start_idx = i * unit_length + j  # 计算起始索引
            end_idx = start_idx + step  # 计算结束索引
            if (start_idx >= total_length):  # 如果起始索引超出总长度，跳出循环
                break
            if (end_idx > total_length):  # 如果结束索引超出总长度，只取到总长度
                end_idx = total_length
            content = str(df['text'].to_list()[start_idx:end_idx])  # 获取文本内容
            messages.append({"role": "user", "content": content})  # 添加用户消息
        messages_list.append(messages)  # 添加到消息列表

    logging.info(f"messages_list = {messages_list}")
    logging.info(f"total_length = {len(messages_list)}")
    # # 调用 API
    list_save_xlsx = []  # 初始化保存结果的列表
    for messages in messages_list:
        logging.info(f"Processing {len(messages)} messages")
        inputs_text = [item for item in messages if item.get("role") == "user"]
        logging.info(f"inputs_text:{inputs_text}")
        # try:
        response = client.chat.completions.create(
            model="soikit_test",  # 指定模型
            messages=messages,  # 传递消息
        )
        logging.info(f"response:{response}")
        content = response.choices[0].message.content  # 获取API响应内容

        logging.info(f"这是gpt4一个批跑出来的结果:{content}")
        list1 = custom_split(content)  # 自定义切割
        logging.info(f"切割后的List1：{list1}")

        analysis_processes = [extract_analysis(item) for item in list1]  # 提取分析
        logging.info(f"anaysis_processes: {analysis_processes}")
        # list_output = [f'{process_text(item)}\n\n{extract_json(item)}' for item in
        #                analysis_processes]  # 处理文本并提取JSON
        # logging.info(f"最终输出结果list_output: {list_output}")
        list_save_xlsx.extend(analysis_processes)  # 添加到结果列表
        # except Exception as e:  # 异常处理
        #     logging.error(f"Error processing messages: {messages}\nException: {e}")  # 记录错误日志
        #     list_input = [item for item in messages if item.get("role") == "user"]  # 获取用户消息
        #     list_demo = [sub_item for item in list_input for sub_item in eval(item.get("content"))]  # 处理用户消息内容
        #     list_save_xlsx.append(len(list_demo) * "Error")  # 添加错误信息

    df['gpt4o_output'] = list_save_xlsx  # 将结果添加到DataFrame的新列中
    return df  # 返回处理后的DataFrame


# 自定义切割函数
def custom_split(content):
    # 检查content是否以"###"开头
    if content.startswith("###"):
        split_points = ["###"]
    else:
        split_points = ["老师话语"]

    result = []  # 初始化结果列表
    temp = content  # 临时存储内容
    for point in split_points:
        parts = temp.split(point)  # 根据切割点进行切割
        result.extend([part for part in parts if part])  # 添加非空部分到结果列表
        # 移除第一个元素，因为它已经被用作切割点
        if result and not result[0]:
            result.pop(0)
    return result  # 返回结果列表


# 主函数
def gpt4o_main(input_df):
    try:
        result_df = process_text_list(input_df)  # 调用处理函数
        return result_df  # 返回结果
    except Exception as e:  # 异常处理
        logging.error(f"Error in main function: {e}")  # 记录错误日志
        return input_df  # 返回输入的DataFrame


# 测试主函数
if __name__ == "__main__":
    df = pd.read_excel("../../../data/data/705_processed.xlsx")  # 读取Excel文件到DataFrame
    result_df = gpt4o_main(df)  # 调用主函数
    result_df.to_excel("../../../data/3有gpt4o分析过程的数据/705_processed.xlsx", index=False)  # 将结果保存到Excel文件中
