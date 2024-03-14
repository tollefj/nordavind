import json
import os

import torch
from flask import Flask, jsonify, render_template, request, session
from transformers import pipeline

app = Flask(__name__, static_url_path="/static")
app.secret_key = os.urandom(100000)

model_name = "models/nordavind-instruct-7b-v0"
pipe = pipeline(
    "text-generation", model=model_name, torch_dtype=torch.float16, device="cuda:1"
)


system_message = """Du er Nordavind, en hjelpsom assistent. Du skal svare på spørsmål og hjelpe brukere med å finne nødvendig informasjon."""


# def make_prompt(instruction, response=""):
#     chat = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": instruction},
#         {"role": "assistant", "content": response},
#     ]
#     return pipe.tokenizer.apply_chat_template(
#         chat,
#         tokenize=False,
#         add_generation_prompt=len(response) == 0,
#     )
def make_prompt(history, instruction, response=""):
    # from the history, extract all "content" strings and make it the instruction
    content = []
    for obj in history:
        content.append(obj["content"])
    content_str = " ".join(content)
    content_str += "\n" + instruction
    content_str = content_str[-3072:]

    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content_str},
        {"role": "assistant", "content": response},
    ]
    return pipe.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def pred(history, instruction, tokens=128, temp=0.75):
    prompt = make_prompt(history, instruction)
    print(prompt)
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

    history = session.get("history", [])
    print(history)
    output = pred(history, text, tokens, temp)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": output})
    session["history"] = history

    return jsonify({"response": output})


# @app.route("/predict", methods=["POST"])
# def predict():
#     text = request.get_json()["text"]
#     tokens = request.get_json()["tokens"]
#     temp = request.get_json()["temp"]
#     output = pred(text, tokens, temp)
#     return jsonify({"response": output})


# route for just receiving text and nothing else
@app.route("/gen", methods=["POST"])
def gen():
    # get the text from the input: "curl -X POST -d "TEXT"
    text = request.get_json()["text"]
    print(text)
    output = pred(text, tokens=128, temp=0.75)
    return output + "\n\n"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("history", None)
    return jsonify({"response": "Session reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
