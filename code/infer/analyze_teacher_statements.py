# analyze_teacher_statements.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

device = "cuda"  # 使用的设备

# 加载模型和分词器
model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-240"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 定义指令
instruction = "分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程。"


# 定义一个函数，接受 List[str, str] 并返回 List[str, str]
def analyze_class_activity(texts: list) -> list:
    results = []
    for text in tqdm(texts):
        input_text = f"### {instruction}\n### 老师话语：{text}\n### 分析过程："
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=1024)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append([text[0], decoded_output])
    return results


if __name__ == "__main__":
    input_texts = [
        "老师说的话1",
        "老师说的话2"
    ]

    # 调用函数进行分析
    results = analyze_class_activity(input_texts)

    # 输出结果
    for result in results:
        print(result)
