import torch
import datasets
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import trl


train_dataset = datasets.load_dataset('soikit/test_cai_data')

args = TrainingArguments(
    num_train_epochs=3,
    output_dir="./test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="adamw_hf",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="steps",
    save_steps=100,  # 每100步保存一次模型
    save_total_limit=1,  # 最多保存1个模型
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    run_name="lomo-imdb",
)


model_id = "/root/autodl-fs/models--qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=1024,
)

trainer.train()




