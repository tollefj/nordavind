import os

import torch
from flask import Flask, jsonify, render_template, request
from transformers import pipeline

app = Flask(__name__, static_url_path="/static")


model_name = "models/nordavind-instruct-7b-v0"
pipe = pipeline(
    "text-generation", model=model_name, torch_dtype=torch.float16, device="cuda:1"
)


system_message = """Du er Nordavind, en hjelpsom assistent. Du skal svare på spørsmål og hjelpe brukere med å finne nødvendig informasjon."""


def make_prompt(instruction, response=""):
    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return pipe.tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=len(response) == 0,
    )


def pred(instruction, tokens=128, temp=0.75):
    prompt = make_prompt(instruction)
    outputs = pipe(
        prompt,
        max_new_tokens=tokens,
        do_sample=True,
        temperature=temp,
        top_k=50,
        top_p=0.1,
        no_repeat_ngram_size=3,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )
    return outputs[0]["generated_text"][len(prompt) :].strip()


@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json()["text"]
    tokens = request.get_json()["tokens"]
    temp = request.get_json()["temp"]
    output = pred(text, tokens, temp)
    return jsonify({"response": output})


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
