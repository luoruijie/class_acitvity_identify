#!/bin/bash

# 提示用户输入输入文件夹和输出文件夹
read -p "请输入输入文件夹的路径: " input_dir
read -p "请输入输出文件夹的路径: " output_dir
read -p "请输入配置文件的路径: " config_file

# 检查输入文件夹和输出文件夹是否存在
if [ ! -d "$input_dir" ]; then
  echo "输入文件夹不存在: $input_dir"
  exit 1
fi

if [ ! -d "$output_dir" ]; then
  echo "输出文件夹不存在: $output_dir"
  exit 1
fi

if [ ! -f "$config_file" ]; then
  echo "配置文件不存在: $config_file"
  exit 1
fi

# 提示用户输入模型号
echo "请输入模型号（用空格分隔）："
read -r -a models

# 确认用户输入的模型号
# shellcheck disable=SC2145
echo "你输入的模型号是: ${models[@]}"

# 遍历输入文件夹中的所有Excel文件并处理
for input_file in "$input_dir"/*.xlsx; do
    if [ -f "$input_file" ]; then
        # shellcheck disable=SC2145
        echo "Processing $input_file with models ${models[@]}"
        python process_files.py --input_file "$input_file" --output_dir "$output_dir" --models "${models[@]}" --config_file "$config_file"
    fi
done
