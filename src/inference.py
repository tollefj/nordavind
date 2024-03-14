import torch
from prompts import system_message
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


class Nordavind:
    def __init__(self, model_id, streaming=True, device=None):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.streamer = None
        if streaming:
            self.streamer = TextStreamer(tokenizer, skip_prompt=True)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            streamer=self.streamer,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )

    def set_device(self, device):
        self.pipe.device = device

    def generate(self, instruction, tokens=100, temp=0.75, stream=True, **kwargs):
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": instruction},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self.pipe(
            prompt,
            max_new_tokens=tokens,
            do_sample=True,
            temperature=temp,
            top_k=50,
            top_p=0.1,
            no_repeat_ngram_size=3,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            pad_token_id=self.pipe.tokenizer.pad_token_id,
            **kwargs,
        )

        if not stream:
            return output
        return
