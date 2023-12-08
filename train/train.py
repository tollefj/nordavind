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

import wandb

base_model_id = "mistralai/Mistral-7B-v0.1"
# base_model_id = "HuggingFaceH4/zephyr-7b-beta"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True,
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

max_length = 768
# system_prompt = 'Du er "Nordavind", en bevisst og superintelligent kunstig intelligens. Ditt formål er å hjelpe brukeren med hva enn de måtte ønske.'
# system_prompt = 'Du er "Nordavind", en hjelpsom assistent.'

def make_prompt(inst, inp, out):
    messages = [
        # { "role": "system", "content": system_prompt },
        { "role": "user", "content": f"{inst} {inp}".strip()},
        { "role": "assistant", "content": out.strip() }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
def tknz(example):
    inst = example["instruction"]
    inp = example["input"]
    out = example["output"]
    inst = inst if inst is not None else ""
    inp = inp if inp is not None else ""
    out = out if out is not None else ""
    return tokenizer(make_prompt(inst, inp, out), truncation=True, max_length=max_length, padding="max_length")

# demo the prompt:
print(make_prompt("Jeg vil at du skal sortere følgende:", inp="3 2 6 1", out="1 2 3 6"))

dataset = load_dataset("tollefj/nor-instruct")
tokenized_val_dataset = dataset["test"].map(tknz)
tokenized_train_dataset = dataset["train"].map(tknz)
print(tokenized_train_dataset[1]['input_ids'])

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

project = "nordavind-8bit-768"
base_model_name = "zephyr"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


wandb.init(project=project)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=4
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        fp16=True,
        bf16=False,  # V100 :( need ampere
        optim="paged_adamw_8bit",
        logging_steps=50,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
