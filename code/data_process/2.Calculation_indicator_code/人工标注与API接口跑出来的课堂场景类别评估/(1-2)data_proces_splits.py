import os

import pandas as pd


def sub_frame(dataset):
    """
    输入: dataset (类型: pandas DataFrame) - 包含至少有 ‘start_time’ 和 ‘end_time’ 列的原始数据集。
    输出: sub_dfs (类型: list of pandas DataFrame) - 包含分割后的子数据集列表。
    描述: 此函数接收一个 DataFrame，根据 ‘start_time’ 和 ‘end_time’ 列的值将数据集分割成多个子数据集。
    如果当前行的 ‘start_time’ 与上一行的 ‘end_time’ 之间的时间差大于3秒，则认为是一个新的对话开始，
    并将该行的索引位置添加到结束位置列表中。最后，根据这些结束位置将原始数据集分割成多个子数据集。
    """
    df = dataset
    end_positions = []
    # 预先加载需要的数据
    start_times = df['start_time'].values
    end_times = df['end_time'].values

    # 遍历所有行，寻找子对话的结束位置
    for i in range(1, len(df)):
        if start_times[i] - end_times[i - 1] > 3000:
            end_positions.append(i)

    # 根据结束位置分割数据集
    sub_dfs = [df.iloc[0:end_positions[0]]]
    sub_dfs.extend([df.iloc[end_positions[i]:end_positions[i + 1]] for i in range(len(end_positions) - 1)])
    sub_dfs.append(df.iloc[end_positions[-1]:])
    return sub_dfs


# 遍历每个sub_dataframe，并根据其'time'列的值来决定是否截取其内容
def process_sub_dataframe(sub_dataframe):
    sub_datasets_list = []
    for item in sub_dataframe:
        # 获取'time_range'列
        time_range_series = item['time_range']
        # 计算总时间是否超过15000毫秒
        total_time = time_range_series.sum()
        # 判断是否超过15000毫秒
        if total_time > 15:
            # 如果最后一行的'time_range'大于15秒
            if time_range_series.iloc[-1] > 15:
                sub_datasets_list.append(item.tail(1))
            else:
                # 从最后一行开始向前遍历，累加'time_range'的值
                cumulative_time = 0
                for i in range(len(time_range_series) - 1, -1, -1):
                    cumulative_time += time_range_series.iloc[i]
                    if cumulative_time > 15:
                        # 找到累加超过15秒的行
                        sub_datasets_list.append(item.iloc[i:])
                        break
        else:
            # 如果小于15000毫秒，则保留全部内容
            sub_datasets_list.append(item)
    return sub_datasets_list


def text_processing(text):
    # 去掉标点符号和空白字符，只保留字母数字字符
    item1 = ''.join(c for c in text if c.isalnum())
    # 将处理过的字符串用单个空格连接，确保中间没有多余空白字符
    item1 = ' '.join(item1.split())
    if len(item1) > 4:
        return text
    else:
        return ''


def convert_milliseconds(ms):
    # 计算总秒数
    ms = int(ms)
    total_seconds = ms // 1000

    # 计算分钟和剩余秒数
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return f"{minutes}分{seconds}秒"


def main(input_file, output_file):
    global columns
    datasets = pd.read_excel(input_file,engine='openpyxl')
    filtered_dataframe = datasets[datasets['label'] == 0].reset_index(drop=True)
    sub_dataframe = sub_frame(filtered_dataframe)
    # 计算每个子数据集的时间范围
    for item in sub_dataframe:
        item['time_range'] = (item['end_time'] - item['start_time']) / 1000
    process_sub_dataframe_list = process_sub_dataframe(sub_dataframe)
    # 遍历处理后的数据，并为每个元素添加一个 'join_text' 键，该键包含合并后的 'text' 列内容
    columns = ['S_T', 'E_T', 'text', 'label']
    empty_dataframe = pd.DataFrame(columns=columns)
    S_T_list = []
    E_T_list = []
    teacher_words_list = []
    for sub_df in process_sub_dataframe_list:
        # 获取第一行的'start_time'值
        S_T = sub_df.iloc[0]['start_time']
        # 获取最后一行的'end_time'值
        E_T = sub_df.iloc[-1]['end_time']
        teacher_words = ' '.join(sub_df['text'])
        S_T_list.append(S_T)
        E_T_list.append(E_T)
        teacher_words_list.append(teacher_words)
    empty_dataframe['S_T'] = S_T_list
    empty_dataframe['E_T'] = E_T_list
    empty_dataframe['S_T'] = empty_dataframe['S_T'].apply(convert_milliseconds)
    empty_dataframe['E_T'] = empty_dataframe['E_T'].apply(convert_milliseconds)
    empty_dataframe['text'] = teacher_words_list
    empty_dataframe['text'] = empty_dataframe['text'].apply(text_processing)
    # 删除 'teacher_words' 列中值为空的行，并重置索引
    empty_dataframe = empty_dataframe[empty_dataframe['text'] != '']
    empty_dataframe.reset_index(drop=True, inplace=True)
    empty_dataframe['text'] = empty_dataframe['text'].apply(lambda x: x.replace(" ", ""))
    print(empty_dataframe)
    empty_dataframe.to_excel(output_file, index=False)


if __name__ == '__main__':
    # 这里是对一个文件夹里面的内容进行批量处理

    data_dir = "C:\\Users\\zonekey008\\Desktop\\class_acitvity_identify\\code\data_process\\1.Multiple_interface_joint_calls\\data\\1原始数据"
    output_dir = "C:\\Users\\zonekey008\\Desktop\\class_acitvity_identify\\code\data_process\\1.Multiple_interface_joint_calls\\data\\2切割后的数据"
    if not os.path.exists(output_dir) or not os.path.exists(data_dir):
        os.makedirs(output_dir)
        os.makedirs(data_dir)

    data_dir_list = os.listdir(data_dir)
    print(data_dir_list)
    for input_file in data_dir_list:
        print("input_file",input_file)
        if input_file.endswith(".xlsx"):
            input_file_path = os.path.join(data_dir, input_file)
            print(input_file_path)
            output_file_path = os.path.join(output_dir, input_file[:-5] + "_processed.xlsx")
            main(input_file_path, output_file_path)

