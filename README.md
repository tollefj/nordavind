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

Trained on a V100 32GB. [See example outputs below!](#examples)

## Prompt format:
```python
<s>{system_prompt} [INST] {inst} [/INST] \\n {res} </s>
```
Inference:
```python
<s>{system_prompt} [INST] {inst} [/INST] \\n *YOUR_PROMPT*
```

## Running the model
Suggested (PEFT adapter):
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE_MODEL = "norallm/normistral-7b-warm"
ADAPTER = "tollefj/nordavind-7b-instruct-adapter"
model = AutoModelForCausalLM.from_pretrained(
    ADAPTER,
    device_map="auto",
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
```
or the full model (will be updated less frequently):
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="tollefj/nordavind-7b-instruct-warm")
```
See [train.py](train/train.py) for training details and [util.py](train/util.py) for the prompt format. Here you can import the `generate(...)` function for inference.
So far, the model has seen better results with greedy sampling.

## TODO:
- [ ] Define more tasks/templates for generation


## Things it can do...
- Translation:
    - "Oversett fra engelsk til norsk"
    - "Oversett frasen ''*tekst*'' til *språk*
- Listing:
    - Lag en liste over ...
    - Ranger en liste med fire elementer fra lavest til høyest...
- Explanations:
    - Beskriv en av fordelene ved...
    - Forklar hvordan ...
    - Kan du tenke på ...
    - Beskriv konseptet ...
    - Hvordan kan ...
- Tasks:
    - Du vil få en oppgave. Du må generere et detaljert og langt svar. ...
    - Bruker vil du gi deg en oppgave.
- Summarization
    - Oppsummer ...
    - Gi en kort oppsummering av ...
    - Lag en kort oppsummering av teksten: ...

## Examples

```python
#https://www.nrk.no/nordland/dayo-i-bodo-fekk-jobb-i-siste-augeblikk-1.16753304
text = "Dayo måtte ut av landet om han ikkje fekk rett jobb på tre veker. Advokat meiner få utanlandske studentar forstår kor viktig val av studie er."

generate(model, tokenizer, text, task="create-title")
```
> Utenlandske studenter må velge riktig studie for å få jobb i Norge

*generate("Lag en skikkelig norsk setning", tokens=50, first_sent=True)*

> Jeg liker å gå på tur i skogen.

*generate("Lag en liste med tre forskjellige bruksområder (inkludert eksempler) for kunstig intelligens.")*

>1. Automatisering av oppgaver: AI kan brukes til å automatisere oppgaver som å svare på kundehenvendelser, behandle medisinske bilder og diagnostisere sykdommer, og til og med kjøre biler.
>2. Forbedre kundeservice: AI kan brukes for å forbedre kundeservicen ved å gi personlige anbefalinger, gi raskere svar og gi mer nøyaktige svar.
>3. Forbedre sikkerhet: AI kan brukes i sikkerhetssystemer for å oppdage og forhindre cyberangrep, oppdage svindel og oppdage mistenkelige aktiviteter.


*generate("Hvem tror du vinner valget i Bergen?", tokens=60)*
>Det er vanskelig å forutsi hvem som vinner valget i Bergen, men det er sannsynlig at enten Arbeiderpartiet eller Høyre vil vinne.  Arbeiderpartiet har hatt makten i Bergen i mange år, men Høyre har hatt en sterk tilstedeværelse i byen de siste årene.

*generate("Forklar dyp læring til en femteklassing")*
>Dyp læring er en type maskinlæring som bruker nevrale nettverk for å lære komplekse mønstre i data. Det innebærer å trene en datamaskin til å gjenkjenne mønstre i data og ta beslutninger basert på disse mønstrene.

