import time
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd


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


def main(input_texts):
    """
    主函数。

    功能:
        执行整个预测流程，包括加载模型、读取数据、处理数据、保存结果等。

    输入:
        input_texts (list of str): 需要处理的文本列表

    输出:
        list of str: 处理后的预测结果列表
    """
    device = "cuda"
    model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-180"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400)
    instruction1 = """分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程。
    """

    llm, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    if not llm or not tokenizer:
        return []

    batch_size = 24
    texts = ["###" + instruction1 + "\n" + "###老师话语：" + text + "\n" + "###分析过程：" for text in input_texts]
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    all_predictions = []
    for batch in batches:
        batch_predictions = process_batch(batch, llm, sampling_params)
        all_predictions.extend([item.outputs[0].text for item in batch_predictions])
    print("all_predictions", all_predictions)
    # all_predictions_processed = [extract_content(item) for item in all_predictions]

    ## 第二步,用instructions2来从中提取出json格式。
    instruction2 = """从给定的文本中提取相关的内容，构建一个JSON对象。

JSON对象结构：

{
  "label": "string",
  "status": "string",
  "key_text": "string"
}
label：填写分析过程中识别的课堂活动类别，如果识别出多个课堂活动类别，用“、”连接多个活动类别，但不要重复填写。
status：填写分析过程中识别的课堂活动的进行状态（如“开始”、“进行中”或“结束”）。
key_text：填写label中第一个课堂活动类别对应的课堂活动指令语句。

如果从分析过程中无法识别出任何预设的课堂活动类别，则所有字段都填写“NA”。

给定的文本如下：

    """
    texts2 = [instruction2 + item for item in all_predictions]

    batches_instruction2 = [texts2[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    all_predictions_2 = []
    for batch in batches_instruction2:
        batch_predictions = process_batch(batch, llm, sampling_params)
        print("batch_predictions", batch_predictions)
        all_predictions_2.extend([item.outputs[0].text for item in batch_predictions])

    all_predictions_processed_2 = [extract_content(item) for item in all_predictions_2]

    print("all_predictions", all_predictions_processed_2)
    return all_predictions, all_predictions_processed_2


if __name__ == '__main__':
    # 示例输入
    df = pd.read_excel("高希娜.xlsx")
    input_texts = df['text'].to_list()
    analysis_predictions, label_predictions = main(input_texts)
    df['Analysis_process'] = analysis_predictions
    df['class_activity_label'] = label_predictions
    df.to_excel("高希娜_vllm_batch.xlsx")
    # for idx, prediction in enumerate(predictions):
    #     print(f"预测结果 {idx + 1}：", prediction)

