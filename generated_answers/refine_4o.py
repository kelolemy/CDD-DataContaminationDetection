import re


def remove_non_code(input_text):
    # Regex to match code blocks and capture the content after any optional language hint
    code_block_pattern = r"```(\w+)?\s*(.*?)```"
    
    # Extract all code blocks and clean them
    matches = re.findall(code_block_pattern, input_text, re.DOTALL)
    
    # Get only the content inside the code block
    transformed_matches = [match[1].strip() for match in matches]
    
    # Join the results with newline
    return "\n".join(transformed_matches)


def strip_and_truncate(text, coding_task=False):
    """
    If coding_task=True, for each function definition in 'text' (including multiple), 
      - identifies the function name between 'def ' and '(' 
      - removes any triple-quoted docstring that immediately follows the function signature
    Otherwise, returns the original text unchanged.

    The docstring is defined as a block of \"\"\"...\"\"\" or '''...''' 
    that appears right after the function signature.

    Args:
        text (str): The entire code or text snippet.
        coding_task (bool): If True, perform removal of docstrings; if False, do nothing.

    Returns:
        str: The updated text, with function docstrings removed if coding_task=True.
    """
    if not coding_task:
        return text

    # Regex: 
    # (1) Capture the function signature (group 1) => "def foo(...) :" including whitespace 
    # (2) Sub-capture the function name (group 2) => used if you want to log or store
    # (3) Capture the triple-quoted docstring (group 3) => We'll remove this part
    pattern = r'(def\s+([A-Za-z_]\w*)\s*\(.*?\):\s*)(["\']{3}[\s\S]*?["\']{3})'

    # We use a while loop to repeatedly remove docstrings from each occurrence, 
    # including multiple functions or multiple docstrings in a row.
    text_out = text
    while True:
        match = re.search(pattern, text_out, flags=re.DOTALL)
        if not match:
            break
        # docstring = match.group(3)  # if you want to see or log it
        prefix = match.group(1)
        # We'll keep the function signature, remove the docstring:
        text_out = re.sub(pattern, prefix, text_out, count=1, flags=re.DOTALL)

    return text_out


import json
with open("humaneval_4o_20_Ps_0.7.json", "r") as f:
    all_data = json.load(f)

refined = []
for data in all_data:
    rows = []
    answer = data['canonical_solution']
    for row in data['generated_answers']:
        if '```' in row:
            row = remove_non_code(row)
            # row = strip_and_truncate(row)
            # if row.startswith('def ') and answer.startswith('def')
            # row = row.split('```python')[1].split('```')[0]
        rows.append(row)
    x = data.copy()
    x['generated_answers'] = rows
    refined.append(x)

with open("refined_4o_all_strip.json", "w") as f:
    json.dump(refined, f, indent=4)

input_1 = "hi it is me ```python x = 1``` and then text, after that `x` defined, it should be ```json{\"name\": \"ka\"} ``` and after all that it is `x` is defined, it should be ```javascript x = 2;``` and then it will be ```print_all()```"
output_1 = "python x = 1\n{\"name\": \"ka\"} \n x = 2;\nprint_all()"