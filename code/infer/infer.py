import time
import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_model_and_tokenizer(model_path, device):
    try:
        llm = LLM(model_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'
        return llm, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def read_file(file_path):
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts = [file.read()]
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            texts = df['text'].tolist()
        else:
            print("Unsupported file format")
            return []
        return texts
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def process_batch(batch_texts, llm, sampling_params):
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
    index = text.find('}')
    return text[:index + 1] if index != -1 else text

def process_text(text, llm, sampling_params):
    print("***************模型接收的输入*******************", text)
    predictions = process_batch([text], llm, sampling_params)
    return predictions[0]

def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False)
        print("Prediction and saving to Excel completed.")
    except Exception as e:
        print(f"Error saving to Excel file: {e}")

def main(file_path, token_decode, model_path):
    device = "cuda"
    if token_decode == "beam_search":
        sampling_params = SamplingParams(temperature=0,
                                         top_p=1,
                                         max_tokens=400,
                                         use_beam_search=True,
                                         skip_special_tokens=False,
                                         best_of=10,
                                         include_stop_str_in_output=True,
                                         early_stopping=True)
    else:
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=0.95,
                                         max_tokens=500,
                                         skip_special_tokens=False,
                                         include_stop_str_in_output=True)

    llm, tokenizer = load_model_and_tokenizer(model_path, device)
    if not llm or not tokenizer:
        return

    texts = read_file(file_path)
    if not texts:
        return

    if file_path.endswith('.txt'):
        prediction = process_text(texts[0], llm, sampling_params)
        prediction_processed = extract_content(prediction.outputs[0].text)
        print("\n")
        print("预测结果：",prediction.outputs[0].text)

    elif file_path.endswith('.xlsx'):
        instruction = """“待分析文本”是一段老师在课堂中说的话，请你预测老师在说完这段话后，学生要开展的课堂活动。课堂活动类别包括：个体发言、个体展示、独立练习、学生齐写、小组学习、学生听写、学生齐读，多人展示，集体未知，个体未知。课堂活动的预测过程为：

	step1：识别与课堂活动相关的特征短语或特征短句，先判定老师说完这段话后，学生要开展的课堂活动类别，再判定该课堂活动的进行状态，分为三个状态：开始、进行中、结束。输出分析过程。

	step2: 输出一个结构化结果，为一个JSON对象，输出前缀为”输出结果：“，该 JSON 对象包含3个属性，定义如下：


	{
	"label": "string",
	"status": "string",
	"key_text": "string"
	}
	其中，“label”填写的是step1中预测的课堂活动类别；如果预测出多个课堂活动类别，则在"label"中用“、”连接各类别；“status”填写的是课堂活动的进行状态；“key_text”填写的是与预测的第一个课堂活动类别对应的课堂活动指令语句。如果分析过程中没有预测任何课堂活动类别，则"label"、"status"和"key_text"字段都填写"NA"。

    
  
	待分析文本："""

        batch_size = 16
        texts_with_instruction = [instruction + text + "###课堂活动的分析过程为:" for text in texts]
        print("text_with_instruction",texts_with_instruction[0])
        batches = [texts_with_instruction[i:i + batch_size] for i in range(0, len(texts_with_instruction), batch_size)]

        all_predictions = []
        for batch in batches:
            batch_predictions = process_batch(batch, llm, sampling_params)
            all_predictions.extend([item.outputs[0].text for item in batch_predictions])

        all_predictions_processed = [extract_content(item) for item in all_predictions]
        df = pd.DataFrame({'text': texts, 'analysis_process': all_predictions_processed})
        save_to_excel(df, "高希娜_预测结果_vllm.xlsx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some texts.')
    parser.add_argument('--file_path', type=str, required=True, help='The path to the input file (txt or xlsx).')
    parser.add_argument('--token_decode', type=str, required=True, help='The method for token decode (beam_search or sampling).')
    parser.add_argument('--model_path', type=str, required=True, help='The path to the model.')
    args = parser.parse_args()
    
    main(args.file_path, args.token_decode, args.model_path)


#python infer.py --file_path prompt.txt --model_path /root/autodl-fs/qwen_7b_GaLore_dataset_5/checkpoint-500/ --token_decode sampling