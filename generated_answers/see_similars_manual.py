import json

with open("humaneval_4o_20_Ps_0.7.json", "r") as f:
    all_outputs = json.load(f)


from collections import Counter

# Filter dictionaries where "generated" contains duplicates
result = [
    item for item in all_outputs
    if any(count > 1 for count in Counter(item["generated_answers"]).values())
]

print(result)