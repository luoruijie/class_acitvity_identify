import re

import pandas as pd
from tqdm import tqdm

with open("config/prompt_file.txt", "r", encoding="utf-8") as f:
    prompt = f.read()


# 提取分析过程内容的函数
def extract_analysis(text):
    start_index = text.find("分析过程")
    if start_index != -1:
        return text[start_index:]
    return ""


def process_text(text):
    """
    处理文本内容。

    功能:
        查找并去除关键词及其后面的所有文本。如果文本不包含任何关键词，将文本按行拆分。

    输入:
        text (str): 输入的文本。

    输出:
        processed_text (str): 处理后的文本。
    """
    # 定义关键词列表
    keywords = ["输出结果","json\n复制代码", "```json", "{", "结构化结果", "输出结果", "输出结果如下：", "输出结构化结果", "JSON",
                "输出:", "输出分析过程：", "JSON 结果：", "[", "JSON结构化结果如下：", "JSON 结构输出：", "JSON对象如下:", "结果：",
                "结果输出", "输出分析结果如下：", "JSON输出：", "分析过程: \n```json", "结果如下：", "输出一个结构化结果：",
                "结构化结果："]

    # 初始化一个变量，用于记录最早出现的关键词的位置
    min_index = len(text)
    min_keyword = None

    # 查找每个关键词在文本中的位置，并记录最早出现的关键词
    for keyword in keywords:
        index = text.find(keyword)
        if 0 <= index < min_index:
            min_index = index
            min_keyword = keyword

    # 如果找到了关键词，则按最早出现的关键词进行分割
    if min_keyword:
        text = text.split(min_keyword)[0]
    else:
        # 如果没有找到任何关键词，将文本按行拆分
        text = text.split("\n")[0]

    # 将换行符替换为空格
    text = text.replace("\n", "")

    return text



def extract_json(text):
    text = re.search(r'\{(?:[^{}]*|\{[^{}]*\})*\}', text).group(0)
    text = text.replace("\n", " ")
    return text


if __name__ == '__main__':
    data = pd.read_excel("combined.xlsx")

    data = data.dropna(axis=0, how='all')
    data = data[~data['predict'].str.contains("error")].reset_index(drop=True)
    gpt4o_output = data['predict'].to_list()

    df = pd.DataFrame()
    df['file_name'] = data['file_name']
    df['text'] = data['text']
    output_analysis = [f'{process_text(item)}' for item in gpt4o_output]
    output_json = [f'{extract_json(item)}' for item in tqdm(gpt4o_output)]
    df['gpt4o_output_analysis'] = output_analysis
    df['gpt4o_output_json'] = output_json
    list1 =[]
    for i,item in tqdm(enumerate(output_json)):
        try:
            list1.append(eval(item).get("label"))
        except Exception as e:
            print(i)

    df['gpt4o_label'] = [eval(item).get("label") for item in tqdm(output_json)]
    df.to_excel("combined_file_splited_new.xlsx", index=False)
