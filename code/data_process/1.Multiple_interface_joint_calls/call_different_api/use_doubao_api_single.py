import os
import pandas as pd
from tqdm import tqdm
import logging
from volcenginesdkarkruntime import Ark
from data_process_public import prompt

# 配置日志
logging.basicConfig(filename='../log/doubao.log', level=logging.INFO, encoding="utf-8", filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 设置API密钥为环境变量
os.environ["ARK_API_KEY"] = "7ee226e5-9cdc-478d-9132-96405749eae6"
print("----- standard request -----")
# 初始化Ark客户端
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


# 定义处理函数的功能：该函数接收一个DataFrame，逐行处理其内容，通过API调用生成相应的输出，并将输出添加到DataFrame中
def doubao_main(df: pd.DataFrame) -> pd.DataFrame:

    # 非流式处理：逐行处理DataFrame中的文本
    for i in tqdm(range(len(df))):
        try:
            # 调用API生成结果
            completion = client.chat.completions.create(
                model="ep-20240715024036-xx49p",  # 指定模型
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": df.loc[i, 'text']},
                ],
            )
            # 打印输出结果
            print("输出结果:",completion.choices[0].message.content)
            # 将结果存入DataFrame的新列中
            df.loc[i, "predict"] = completion.choices[0].message.content

        except Exception as e:
            # 捕获异常并记录日志
            logging.error(f"Error processing row {i}: {e}")
            # 在DataFrame中记录错误
            df.loc[i, 'predict'] = "error"

    return df


if __name__ == '__main__':
    df = pd.read_excel("data/726_processed.xlsx")
    df = doubao_main(df)
    df.to_excel("data/726_processed.xlsx")
