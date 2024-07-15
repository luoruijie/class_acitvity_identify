import os
import pandas as pd




# 定义文件名列表
file_names = os.listdir("../data")
# 定义统一的列顺序


common_columns = ['S_T', 'E_T', 'text', 'label', 'qwen_long_output', 'gpt4o_output', 'doubao_output_new']  # 用你实际的列名替换这里
# 读取并合并文件
combined_df = pd.DataFrame()
for file in file_names:
    df = pd.read_excel("data/" + file)  # 读取 Excel 文件
    # 重新排列列的顺序
    df = df[common_columns]
    # 将当前文件的数据追加到合并后的 DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df = combined_df[~combined_df['gpt4o_output'].str.contains("error")]
combined_df.to_excel('combined_file.xlsx', index=False)
