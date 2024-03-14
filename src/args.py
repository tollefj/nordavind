from peft import LoraConfig
from transformers import TrainingArguments


# LoRA config based on QLoRA paper & Sebastian Raschka experiment
def get_peft_config():
    return LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def get_training_args(output_dir):
    return TrainingArguments(
        # output_dir="models/nordavind-7b-warm-v0",
        output_dir=output_dir,
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
    )
