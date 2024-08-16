# 导入需要的包
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_proces_splits import main
# 从对应的文件中
from data_process_public import process_text, extract_json

# 1-2：切割原始文件
data_dir = "data\\1原始数据"
output_dir = "data\\2切割后的数据"
if not os.path.exists(output_dir) or not os.path.exists(data_dir):
    os.makedirs(output_dir)
    os.makedirs(data_dir)

data_dir_list = os.listdir(data_dir)
print(data_dir_list)
for input_file in data_dir_list:
    # print("input_file", input_file)
    if input_file.endswith(".xlsx"):
        input_file_path = os.path.join(data_dir, input_file)
        print(input_file_path)
        output_file_path = os.path.join(output_dir, input_file[:-5] + "_processed.xlsx")
        main(input_file_path, output_file_path)
# 2-3：调gpt4o_api接口两次
# 定义文件夹路径
folder1 = 'data\\2切割后的数据'
folder2 = 'data\\3调api跑出的数据\\调api跑出的数据_first'
folder3 = 'data\\3调api跑出的数据\\调api跑出的数据_second'
folder4 = 'data\\3调api跑出的数据\\调api跑出的数据_old'
# 获取文件夹中的文件列表
files_folder1 = set([item.split("_processed")[0] for item in os.listdir(folder1)])
files_folder_new = set(["_".join(item.split("_processed_output.xlsx")[0].split("_")[1:]) for item in os.listdir(folder2)])
list_files = []
for item in os.listdir(folder4):
    if "first" in item:
        list_files.append("_".join(item.split("_processed_output_first.xlsx")[0].split("_")[1:]))
    elif "second" in item:
        list_files.append("_".join(item.split("_processed_output_second.xlsx")[0].split("_")[1:]))
    else:
        list_files.append("_".join(item.split("_processed_output.xlsx")[0].split("_")[1:]))

files_old = set(list_files)
files_folder2 = files_folder_new.union(files_old)
# 找出只在文件夹1中的文件
different = list(files_folder1 - files_folder2)
print(different)

import subprocess

# 循环调用列表different中的元素，并执行command1
for file in different:
    print(file)
    command1 = f'python main.py --input_file {folder1}\\{file}_processed.xlsx --output_dir {folder2} --models 3 --config_file config/config.json'
    command2 = f'python main.py --input_file {folder1}\\{file}_processed.xlsx --output_dir {folder3} --models 3 --config_file config/config.json'
    print(command1)
    print(command2)
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)


#4.合并数据

def combine_excel_files(folder, output_file):
    # 获取文件夹下所有的Excel文件
    excel_files = [f for f in os.listdir(folder) if f.startswith('gpt4o')]
    # 初始化一个空的DataFrame用于存储所有文件的数据
    combined_df = pd.DataFrame()

    # 遍历所有Excel文件并将它们的内容添加到combined_df中
    for file in excel_files:
        file_path = os.path.join(folder, file)
        df = pd.read_excel(file_path)
        df["file_name"] = file
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 将合并后的DataFrame保存到一个新的Excel文件中

    combined_df.to_excel(output_file, index=False)


combine_excel_files(folder2, "combined_first.xlsx")
combine_excel_files(folder3, "combined_second.xlsx")
#
#
# 5.提取分析过程中的label，并入到gpt4o_label这一列下。
def process_excel(input_file: str, output_file: str):
    # 读取Excel文件
    data = pd.read_excel(input_file)
    # 删除所有为NaN的行
    data = data.dropna(subset=['predict'])
    # 去除'predict'列中包含"error"的行
    data = data[~data['predict'].str.contains("error")].reset_index(drop=True)
    # 将'predict'列转换为列表
    gpt4o_output = data['predict'].to_list()

    # 创建新的DataFrame
    df = pd.DataFrame()
    df['file_name'] = data['file_name']
    df['text'] = data['text']

    # 对gpt4o_output进行分析并提取json
    output_analysis = [f'{process_text(item)}' for item in gpt4o_output]
    output_json = []
    for i, item in tqdm(enumerate(gpt4o_output)):
        try:
            # extract_json(item)
            output_json.append(extract_json(item))
        except Exception as e:
            output_json.append('{"label":"error"}')
    #
    # 添加分析结果到DataFrame
    df['gpt4o_output_analysis'] = output_analysis
    df['gpt4o_output_json'] = output_json

    # 提取标签并添加到DataFrame
    list1 = []
    for i, item in tqdm(enumerate(output_json)):
        try:
            list1.append(eval(item).get("label"))
        except Exception as e:
            print(f"Error at index {i}: {e}")
            list1.append("error")
    df['gpt4o_label'] = list1

    # df['gpt4o_label'] = [eval(item).get("label") for item in tqdm(output_json)]

    # 将结果保存为新的Excel文件
    df.to_excel(output_file, index=False)

process_excel("combined_first.xlsx", "combined_file_splited_new_first.xlsx")
process_excel("combined_second.xlsx", "combined_file_splited_new_second.xlsx")

# 处理两次接口跑出来的dataframe，如果有gpt4o_label不一致数据，则去除，并最终只保存label一致的数据。

# 创建DataFrame
combined_first = pd.read_excel("combined_file_splited_new_first.xlsx")
combined_second = pd.read_excel("combined_file_splited_new_second.xlsx")
# 将combined_second下的gpt4o_label赋值 给combined_first中gpt4o_label_second列


for i in tqdm(range(len(combined_first)),desc = "互相赋值"):
    for j in range(len(combined_second)):
        if combined_first.loc[i, 'text'] == combined_second.loc[j, 'text']:
            combined_first.loc[i, 'gpt4o_label_second'] = combined_second.loc[j, 'gpt4o_label']
# 提取gpt4o_label不一致的行索引
combined_first['gpt4o_label'] = combined_first['gpt4o_label'].replace(np.nan, 0)
combined_first['gpt4o_label_second'] = combined_first['gpt4o_label_second'].replace(np.nan, 0)
unmatch_index = combined_first[combined_first['gpt4o_label']!=combined_first['gpt4o_label_second']].index

# 保留gpt4o_label一致的数据

matched_data = combined_first[~combined_first.index.isin(unmatch_index)].reset_index(drop=True)
#将gpt4o_label列中值 为”error"的行去掉
matched_data = matched_data[matched_data['gpt4o_label']!="error"].reset_index(drop=True)
#将"gpt4o_label"列中值为0替换为"其他"
matched_data['gpt4o_label']= matched_data['gpt4o_label'].replace(0,"其他")
matched_data = matched_data.drop(['gpt4o_label_second'],axis=1)
# 将结果存储为Excel文件
output_path = f"combined_file_splited_new_{len(matched_data)}.xlsx"
matched_data.to_excel(output_path, index=False)
#

# 6.生成最终的训练数据
# 读取两个xlsx文件
# file1 = 'combined_file_splited_old.xlsx'
# file2 = 'combined_file_splited_new.xlsx'
#
# # 使用pandas读取xlsx文件
# df1 = pd.read_excel(file1)
# df2 = pd.read_excel(file2)
#
# # 合并两个DataFrame
#
# df_combined = pd.concat([df1, df2])
# #将gpt4o_label列中值 为”error"的行去掉
# df_combined = df_combined[df_combined['gpt4o_label']!="error"].reset_index(drop=True)
# #将"gpt4o_label"列中值为0替换为"其他"
# df_combined['gpt4o_label']= df_combined['gpt4o_label'].replace(0,"其他")
# df_combined = df_combined.drop(['gpt4o_label_second'],axis=1)
# # 保存合并后的文件
# output_file = f'combined_file_splited_{len(df_combined)}.xlsx'
# df_combined.to_excel(output_file, index=False)
