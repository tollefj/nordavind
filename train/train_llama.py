from datetime import datetime

import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import wandb

base_model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    # llm_int8_has_fp16_weight=True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="right",
    # added these manually in traiing data
    # add_eos_token=True,
    # add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token
max_length = 1024
# system_prompt = 'Du er "Nordavind", en bevisst og superintelligent kunstig intelligens. Ditt formål er å hjelpe brukeren med hva enn de måtte ønske.'

system_prompt = 'Du er "Nordavind", en hjelpsom assistent.'

def tknz(example):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
dataset_id = "tollefj/nor-instruct-v2-llama2"
dataset = load_dataset(dataset_id)
tokenized_val_dataset = dataset["test"].map(tknz)
tokenized_train_dataset = dataset["train"].map(tknz)
print(tokenized_train_dataset[1]['input_ids'])

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    # leaving r and alpha to defaults
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# if torch.cuda.device_count() > 1: # If more than 1 GPU
# model.is_parallelizable = True
# model.model_parallel = True

project = "nollama2-7b-chat"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
wandb.init(project=project, name=run_name)
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=3000,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=False,  # V100 :( need ampere
        # optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this res if you don't want to use weights & baises
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
