from vllm import LLM, SamplingParams
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


def process_text(text, llm, sampling_params):
    """
    处理单个文本并返回预测结果。

    输入:
        text (str): 输入文本
        llm: 语言模型
        sampling_params: 采样参数

    输出:
        预测结果
    """
    batch = [text]
    predictions = process_batch(batch, llm, sampling_params)
    return predictions[0]

def main(input_text: str):
    """
    主函数。

    功能:
        执行整个预测流程，包括加载模型、读取提示、处理数据、保存结果等。

    输入:
        input_text (str): 需要处理的文本

    输出:
        无
    """
    device = "cuda"
    model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-240"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400)



    llm, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    if not llm or not tokenizer:
        return

    # text = f"### {instruction1}\n### 老师话语：{input_text}\n### 分析过程："
    
    # 处理单个文本
    prediction = process_text(input_text, llm, sampling_params)
    prediction_processed = extract_content(prediction.outputs[0].text)

    # 打印或保存结果
    print("预测结果：", prediction_processed)



if __name__ == "__main__":
      # 读取prompt.txt文件内容
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        instruction1 = file.read().strip()
    main(instruction1)
