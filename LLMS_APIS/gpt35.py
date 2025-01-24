import time
from openai import OpenAI
from config import LLM_API_KEY


def call_gpt35(prompt, temperature):
    try:
        client = OpenAI(api_key=LLM_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4096
        )

        llm_answer = ' '.join(response.choices[0].message.content.splitlines())
        return llm_answer
    except Exception as e:
        time.sleep(0.5)
        return call_gpt35(prompt, temperature)
