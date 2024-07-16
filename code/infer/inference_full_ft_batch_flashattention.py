import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

device = "cuda"  # the device to load the model onto

# Load the model and tokenizer
model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-180"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
).to(device)
print("Is the model on GPU?", next(model.parameters()).device == torch.device('cuda'))

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.padding_side = 'left'  # Ensure padding is on the left side

instruction = """分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程,课堂活动类别包括：个体发言、个体展示、独立练习、学生齐写、小组学习、学生听写、学生齐读，多人展示，集体未知，个体未知。

"""

# Read the Excel file
df = pd.read_excel("高希娜.xlsx")

# Function to process a single batch
def process_batch(batch_texts):
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Start measuring time and GPU memory
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()
    
    # Generate predictions
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        # no_repeat_ngram_size=2,
        # temperature=0.9,
        # top_k=50,
        # top_p=0.95,
        # eos_token_id=tokenizer.eos_token_id,  # Ensure the use of the end of sequence token
        # pad_token_id=tokenizer.pad_token_id,  # Ensure proper padding
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    end_memory = torch.cuda.memory_allocated()
    
    # Calculate the memory used during generation
    memory_usage = end_memory - start_memory
    
    print(f"Batch processing time: {elapsed_time:.2f} seconds")
    print(f"Memory used during generation: {memory_usage / (1024 ** 2):.2f} MB")

    # Decode the predictions and ensure <im_end> is included if expected
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return predictions

# Split the data into batches
batch_size = 16
texts = [
    "###" + instruction + "\n" + "###老师话语：" + df.loc[i, 'text'] + "\n" + "###分析过程："
    for i in range(len(df))
]
batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

# Process each batch and collect results
all_predictions = []
for batch in batches:
    batch_predictions = process_batch(batch)
    all_predictions.extend(batch_predictions)

# Save the predictions to the DataFrame
df['full_FT_qwen2_7b_predict'] = all_predictions

# Save the DataFrame to an Excel file
df.to_excel("高希娜_预测结果.xlsx", index=False)

print("Prediction and saving to Excel completed.")
