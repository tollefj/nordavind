system_message = """Du er Nordavind, en hjelpsom assistent. Du skal svare på spørsmål og hjelpe brukere med å finne nødvendig informasjon."""


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["INSTRUCTION"]},
            {"role": "assistant", "content": sample["RESPONSE"]},
        ]
    }
