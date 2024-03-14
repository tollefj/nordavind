<div>
    <h1 align="center">Nordavind-7B</h1>
    <p align="center">
    <img width="250" src="assets/nordavind.jpeg">
    </p>
    <p align="center">
    <em>nordavind - a cold wind from the north</em>
    </p>
</div>

A model trained on top of [normistral-7b (warm)](https://huggingface.co/norallm/normistral-7b-warm) on various Norwegian instruct data with a context length of 2048. The name is derived from [Mistral](https://en.wikipedia.org/wiki/Mistral_(wind)).

Trained with an RTX 4090 for approx 10 hours (2913 steps).

Uses the [ChatML format](https://huggingface.co/docs/transformers/main/en/chat_templating):

```python
"messages": [
    {"role": "system", "content": system_message},
    {"role": "user", "content": sample["INSTRUCTION"]},
    {"role": "assistant", "content": sample["RESPONSE"]},
]
```

## Running the model
See the simplified inference example in [src/nordavind_inference.ipynb](src/nordavind_inference.ipynb).
