import time
from openai import OpenAI
from config import LLM_API_KEY


def call_gpt4o(prompt, image_base64, temperature):
    """
    Calls GPT-4o with optional base64-encoded image data appended to the user message.
    
    Args:
        prompt (str): The user's text prompt or question.
        temperature (float): Sampling temperature.
        image_base64 (str): A base64 string, e.g. "data:image/jpeg;base64,/9j/4AAQSk..."
                            If present, it is appended to the user content.

    Returns:
        str: The GPT-4o response text.
    """
    try:
        client = OpenAI(api_key=LLM_API_KEY)
        sys_msg = "Give the exact answer only. Just the short answer for the question. If it is a multiple choice question, just give the label of the correct answer. No explanations, No steps, No examples, No calculations, No derivation."

        user_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_base64}
                })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=4096
        )

        llm_answer = ' '.join(response.choices[0].message.content.splitlines())
        return llm_answer

    except Exception as e:
        # A simple retry mechanism
        print("Sleeping for 0.5 s")
        print(e)
        time.sleep(0.5)
        return call_gpt4o(prompt, temperature, image_base64)
