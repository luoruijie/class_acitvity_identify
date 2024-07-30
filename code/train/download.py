import os
import argparse
from transformers import AutoModel, AutoTokenizer


def main(model_name, cache_dir):
    # 设置下载目录
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    # 下载模型和分词器
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"模型 '{model_name}' 和分词器已下载到 '{cache_dir}' 目录。")


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='下载模型和分词器')

    # 添加参数
    parser.add_argument('--model_name', type=str, required=True, help='要下载的模型名称')
    parser.add_argument('--cache_dir', type=str, required=True, help='模型缓存目录')

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    main(args.model_name, args.cache_dir)
# python download_model.py --model_name hfl/llama-3-chinese-8b-instruct-v3 --cache_dir /root/autodl-fs
