import torch, os
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# 安装 flash_attention 库
os.system('pip install flash_attn')

# 配置计算数据类型为 bfloat16 和注意力实现方式为 flash_attention_2
compute_dtype = torch.bfloat16
attn_implementation = 'flash_attention_2'

# 模型路径
model_name = "/root/autodl-fs/models--qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# 设置填充位置为左边
tokenizer.padding_side = 'left'

# 加载数据集
# ds = load_dataset("timdettmers/openassistant-guanaco")
ds = load_dataset('soikit/cai_dataset_3_text_coloumn')

# 加载模型，使用 bfloat16 数据类型，并配置设备和注意力实现方式
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=attn_implementation
)

# 启用梯度检查点，减少显存使用
model.gradient_checkpointing_enable()

# 配置模型的填充标记 ID
model.config.pad_token_id = tokenizer.pad_token_id
# 禁用缓存以确保与梯度检查点兼容
model.config.use_cache = False

# 配置训练参数
training_arguments = TrainingArguments(
    output_dir="/root/autodl-fs/qwen_7b_GaLore",  # 模型输出路径
    evaluation_strategy="steps",  # 评估策略为每隔一定步数进行评估
    do_eval=True,  # 启用评估
    per_device_train_batch_size=4,  # 每个设备的训练批次大小
    per_device_eval_batch_size=4,  # 每个设备的评估批次大小
    log_level="debug",  # 日志级别
    optim="galore_adamw",  # 优化器选择 GaLore AdamW
    optim_args="rank=512, update_proj_gap=200, scale=1.8",  # 优化器参数配置
    optim_target_modules=[r".*attn.*", r".*mlp.*"],  # 优化器目标模块
    save_strategy = 'steps',  # 保存策略为每隔一定步数保存一次模型
    save_steps=60,  # 每 60 步保存一次模型
    save_total_limit=1,  # 只保留一个最佳模型
    logging_steps=85,  # 每 85 步记录一次日志
    learning_rate=1e-5,  # 学习率
    eval_steps=10,  # 每 10 步进行一次评估
    bf16=torch.cuda.is_bf16_supported(),  # 是否支持 bfloat16
    num_train_epochs=10,  # 训练的总 epoch 数
    warmup_ratio=0.1,  # 预热比例
    lr_scheduler_type="linear",  # 学习率调度器类型为线性
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    metric_for_best_model="loss",  # 使用 loss 作为选择最佳模型的指标
)

# 创建训练器 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],  # 训练数据集
    eval_dataset=ds['test'],  # 评估数据集
    dataset_text_field="text",  # 数据集中用于训练的文本字段
    max_seq_length=512,  # 最大序列长度
    tokenizer=tokenizer,  # 分词器
    args=training_arguments,  # 训练参数
)

# 开始训练
trainer.train()
