from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

device = "cuda"  # the device to load the model onto

df = pd.read_excel("高希娜.xlsx")
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-fs/qwen_7b_GaLore/checkpoint-180",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/qwen_7b_GaLore/checkpoint-180")

instruction1 = """分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程,课堂活动类别包括：个体发言、个体展示、独立练习、学生齐写、小组学习、学生听写、学生齐读，多人展示，集体未知，个体未知。"""

instruction2 = """请从以下文本中提取信息，并构建一个JSON对象。JSON对象结构如下
{
    "label": "string",
    "status": "string",
    "key_text": "string"
}
具体要求：
    label：填写识别出的课堂活动类别。如果有多个类别，用“、”连接，但不要重复。
    status：填写课堂活动的状态（如“开始”、“进行中”或“结束”）。
    key_text：填写label中第一个活动类别对应的课堂活动指令语句。
    如果无法识别任何课堂活动类别，所有字段都填写“NA”。

"""
for i in range(len(df)):
    messages = [
        {"role": "system", "content": instruction1},
        {"role": "user", "content": text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    df.loc[i, 'Analysis_process'] = response

for i in range(len(df)):
    messages = [
        {"role": "system", "content": instruction2},
        {"role": "user", "content": df.loc[i, 'Analysis_process']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    df.loc[i, 'class_activity_label'] = response

df.to_excel("高希娜_transforerms.xlsx", index=False)
