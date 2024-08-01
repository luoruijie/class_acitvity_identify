
import torch
from modelscope import snapshot_download
import os
model_dir = snapshot_download('qwen/Qwen1.5-32B-Chat', cache_dir='/root/autodl-fs', revision='master')