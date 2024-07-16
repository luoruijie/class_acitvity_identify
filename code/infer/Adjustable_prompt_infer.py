import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser(description="Load model and read prompt for inference.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
    
    args = parser.parse_args()
    
    device = "cuda" # the device to load the model onto
    
    model_name_or_path = args.model_name_or_path
    prompt_file = args.prompt_file
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        instruction = f.read()
    
    print("instruction:", instruction)
    
    inputs = tokenizer.encode(instruction, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=1024)
    
    print("outputs:", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()


# 命令行使用：python Adjustable_prompt_infer.py --model_name_or_path /root/autodl-fs/qwen_7b_GaLore/checkpoint-180 --prompt_file prompt.txt





