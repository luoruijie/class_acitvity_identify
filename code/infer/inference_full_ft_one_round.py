from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"

#推理
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 
model_name_or_path = "/root/autodl-fs/qwen_7b_GaLore/checkpoint-180" 

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# instruction = """分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程,课堂活动类别包括：个体发言、个体展示、独立练习、学生齐写、小组学习、学生听写、学生齐读，多人展示，集体未知，个体未知。"""
instruction = "分析给定的老师话语，写出老师说完这段话后，学生要开展的课堂活动类别的分析过程。"




df = pd.read_excel("高希娜.xlsx")
print("df",df)
for i in tqdm(range(len(df))):
    # text = "###分析给定的老师话语,预测在老师说完这段话后，学生要开展的课堂活动。 ###" +df.loc[i,'text'] +"###分析过程："
    text = "###" +instruction +"\n"+ "###老师话语："+ df.loc[i,'text'] + "\n" + "###分析过程：" 

    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    print("inputs",inputs)
    outputs = model.generate(inputs,max_length=1024)
    print(tokenizer.decode(outputs[0]))
    df.loc[i,'full_FT_qwen2_7b_predict_one'] = tokenizer.decode(outputs[0])

df.to_excel("高希娜.xlsx",index=False)