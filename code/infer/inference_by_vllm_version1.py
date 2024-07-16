import time
import torch
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_model_and_tokenizer(model_path, device):
    """
    加载模型和分词器。

    功能:
        加载指定路径的模型和分词器，并设置分词器的填充方式。

    输入:
        model_path (str): 模型的路径。
        device (str): 设备类型（如 "cuda"）。

    输出:
        llm (LLM): 加载的LLM模型对象。
        tokenizer (AutoTokenizer): 加载的分词器对象。
    """
    try:
        llm = LLM(model_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'
        return llm, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None


def read_excel_file(file_path):
    """
    读取Excel文件。

    功能:
        读取指定路径的Excel文件并返回一个DataFrame对象。

    输入:
        file_path (str): Excel文件的路径。

    输出:
        df (DataFrame): 读取的DataFrame对象。
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def process_batch(batch_texts, llm, sampling_params):
    """
    处理一批文本数据。

    功能:
        对一批文本数据进行处理，生成模型的预测结果，并记录处理时间和内存使用情况。

    输入:
        batch_texts (list of str): 一批需要处理的文本数据。
        llm (LLM): 已加载的LLM模型对象。
        sampling_params (SamplingParams): 采样参数对象。

    输出:
        outputs (list of dict): 模型生成的预测结果。
    """
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()

    try:
        outputs = llm.generate(batch_texts, sampling_params)
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return []

    elapsed_time = time.time() - start_time
    end_memory = torch.cuda.memory_allocated()
    memory_usage = end_memory - start_memory

    print(f"Batch processing time: {elapsed_time:.2f} seconds")
    print(f"Memory used during generation: {memory_usage / (1024 ** 2):.2f} MB")

    return outputs


def extract_content(text):
    """
    提取文本中的JSON对象。

    功能:
        从文本中提取第一个JSON对象，如果没有找到JSON对象则返回原文本。

    输入:
        text (str): 输入的文本。

    输出:
        extracted_text (str): 提取的JSON对象或原文本。
    """
    index = text.find('}')
    return text[:index + 1] if index != -1 else text


def save_to_excel(df, file_path):
    """
    将DataFrame保存为Excel文件。

    功能:
        将DataFrame保存到指定路径的Excel文件中。

    输入:
        df (DataFrame): 要保存的DataFrame对象。
        file_path (str): 目标Excel文件的路径。

    输出:
        无
    """
    try:
        df.to_excel(file_path, index=False)
        print("Prediction and saving to Excel completed.")
    except Exception as e:
        print(f"Error saving to Excel file: {e}")


def main():
    """
    主函数。

    功能:
        执行整个预测流程，包括加载模型、读取数据、处理数据、保存结果等。

    输入:
        无

    输出:
        无
    """
    device = "cuda"
    model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-120"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400)
    instruction1 = """分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程，
    """

    llm, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    if not llm or not tokenizer:
        return

    df = read_excel_file("高希娜.xlsx")
    if df is None:
        return

    batch_size = 24
    texts = ["###" + instruction1 + "\n" + "###老师话语：" + df.loc[i, 'text'] + "\n" + "###分析过程：" for i in
             range(len(df))]
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    all_predictions = []
    for batch in batches:
        batch_predictions = process_batch(batch, llm, sampling_params)
        all_predictions.extend([item.outputs[0].text for item in batch_predictions])

    all_predictions_processed = [extract_content(item) for item in all_predictions]
    df['full_FT_qwen2_7b_predict'] = all_predictions_processed

    save_to_excel(df, "高希娜_预测结果_vllm.xlsx")


if __name__ == '__main__':
    main()
