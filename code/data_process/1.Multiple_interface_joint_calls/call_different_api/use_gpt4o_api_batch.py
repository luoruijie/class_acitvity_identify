import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key='0b63f17479414114b7006e9d1ed56f87',
    api_version="2024-07-01-preview",
    azure_endpoint='https://soikit-gpt4-mini.openai.azure.com/'
)

# Upload a file with a purpose of "batch"
file = client.files.create(
    file=open("../test.jsonl", "rb"),
    purpose="batch"
)

print(file.model_dump_json(indent=2))
file_id = file.id

# Wait until the uploaded file is in processed state
import time
import datetime

status = "pending"
while status != "processed":
    time.sleep(15)
    file_response = client.files.retrieve(file_id)
    status = file_response.status
    print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")