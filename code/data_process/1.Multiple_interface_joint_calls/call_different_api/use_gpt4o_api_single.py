import pandas as pd
from openai import AzureOpenAI
from data_process_public import prompt
import logging

# 配置日志记录
logging.basicConfig(filename='gpt4o_process.log', level=logging.INFO, encoding="utf-8", filemode='w',
                    format='%(asctime)s:%(levelname)s:%(message)s')


def main_gpt4o(df):
    client = AzureOpenAI(
        azure_endpoint="https://zonekey-gpt4o.openai.azure.com/",
        api_key="b01e3eb073fe43629982b30b3548c36e",
        api_version="2024-02-01"
    )

    for i in range(len(df)):
        try:
            response = client.chat.completions.create(
                model="soikit_test",  # model = "deployment_name".
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": df.loc[i, 'text']},
                ]
            )
            output = response.choices[0].message.content
            df.loc[i, 'predict'] = output
            logging.info(f"Row {i} processed successfully. Output: {output}")

        except Exception as e:
            df.loc[i, 'predict'] = "error"
            logging.error(f"Error processing row {i}: {e}")

    return df


if __name__ == "__main__":
    try:
        df = pd.read_excel("data/705_processed.xlsx")
        logging.info("Excel file loaded successfully.")

        df = main_gpt4o(df)

        df.to_excel("data/705_processed.xlsx", index=False)
        logging.info("Excel file saved successfully.")

    except Exception as e:
        logging.error(f"Error in main process: {e}")
