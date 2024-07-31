import json
from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

model_path = ""  # 这是要写死的


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def infer_single(text, model, tokenizer, device, max_new_tokens=400):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def infer_batch(texts, model, tokenizer, device, batch_size=8, max_new_tokens=400):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results


def process_texts(texts, model, tokenizer, device, infer_method):
    instruction1 = """###分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程\n###老师话语："""
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

    if infer_method == 'single':
        analysis_process = []
        for text in texts:
            try:
                analysis_process.append(infer_single(instruction1 + text + "\n###分析过程：", model, tokenizer, device))
            except Exception as e:
                analysis_process.append("ERROR: " + str(e))

        class_activity_label = []
        for analysis in analysis_process:
            try:
                class_activity_label.append(infer_single(instruction2 + analysis, model, tokenizer, device))
            except Exception as e:
                class_activity_label.append("ERROR: " + str(e))
    else:
        texts = [instruction1 + item + "\n###分析过程：" for item in texts]
        try:
            analysis_process = infer_batch(texts, model, tokenizer, device)
        except Exception as e:
            analysis_process = ["ERROR: " + str(e)] * len(texts)

        text2 = [instruction2 + item for item in analysis_process]
        try:
            class_activity_label = infer_batch(text2, model, tokenizer, device)
        except Exception as e:
            class_activity_label = ["ERROR: " + str(e)] * len(text2)

    return analysis_process, class_activity_label


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    if 'text' not in data or not isinstance(data['text'], list):
        return jsonify({"error": "Invalid input, 'text' key must be present and must be a list of strings."}), 400

    texts = data['text']

    infer_method = data.get('infer_method', 'batch')

    try:
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        analysis_process, class_activity_label = process_texts(texts, model, tokenizer, device, infer_method)
    except Exception as e:
        return jsonify({"error": "Model loading or processing error: " + str(e)}), 500

    return jsonify({
        "analysis_process": analysis_process,
        "class_activity_label": class_activity_label
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
