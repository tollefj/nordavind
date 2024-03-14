import torch
from args import get_peft_config, get_training_args
from data import get_data
from trl import SFTTrainer

from llm import init_llm

print(torch.__version__)
assert (
    torch.cuda.get_device_capability()[0] >= 8
), "Hardware not supported for Flash Attention"

model_id = "norallm/normistral-7b-warm"
model, tokenizer = init_llm(model_id)


dataset = get_data()
max_seq_length = 3072

trainer = SFTTrainer(
    model=model,
    args=get_training_args(),
    train_dataset=dataset,
    peft_config=get_peft_config(),
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        # We template with special tokens
        "add_special_tokens": False,
        # No need to add additional separator token
        "append_concat_token": False,
    },
)
trainer.train()
trainer.save_model()
