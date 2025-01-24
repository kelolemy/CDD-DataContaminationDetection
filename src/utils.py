import json
import re
import logging
import nltk
import tiktoken
from transformers import AutoTokenizer
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from config import EMBEDDINGS_OPENAI_API_KEY, CHROMADB_IP, CHROMADB_PORT

##############################################################################
# 1. API CALLS
##############################################################################

# Replace with your actual LLM import
from LLMS_APIS.gpt4o import call_gpt4o as call_api
# from LLMS_APIS.gpt35 import call_gpt35 as call_api

def call_api_wrapper(prompt, temperature=0.7, n=1):
    """
    Wraps the call_api function to request multiple outputs concurrently.
    """
    # with ThreadPoolExecutor(max_workers=min(n, 2)) as executor:
    with ThreadPoolExecutor(max_workers=min(n, 20)) as executor:
        futures = [executor.submit(call_api, prompt, temperature) for _ in range(n)]
        outputs = [f.result() for f in futures]
    return outputs

def call_llm_get_greedy(prompt):
    return call_api(prompt, temperature=0.0)

##############################################################################
# 2. LOAD DATASET
##############################################################################

def prepare_dataset(dataset_name, config_split=None, dataset_to_use=None,
                    prompt_key=None, answer_key=None):
    """
    Prepares a dataset for further processing by loading and returning the specified split,
    along with the keys for prompt and answer fields.
    
    You'll be prompted to give (if needed): `config_split`, `dataset_to_use`, `prompt_key`, `answer_key`.

    Args:
        dataset_name (str): The name of the dataset to load, typically from a data hub
            (e.g., Hugging Face). This can also be a local path if needed.
        config_split (str, optional): The configuration or split to load (e.g., "train",
            "validation", or a subset identifier). Defaults to None, indicating no special split.
        dataset_to_use (str, optional): The key for the specific subset of the dataset,
            such as "test", "train", etc. Defaults to None, which may refer to the first or
            only subset in the loaded dataset.
        prompt_key (str, optional): The name of the field/column representing the prompt
            or question. Defaults to None, meaning a default or best-guess column may be used.
        answer_key (str, optional): The name of the field/column representing the answer
            or solution. Defaults to None, meaning a default or best-guess column may be used.

    Returns:
        tuple: A tuple containing three items:
            1. ds_part (Dataset): The requested portion of the loaded dataset, as dictated
               by `dataset_to_use`.
            2. prompt_key (str): The prompt field name to be used for subsequent processing.
            3. answer_key (str): The answer field name to be used for subsequent processing.
    """
    try:
        if config_split:
            ds = load_dataset(dataset_name, config_split)
        else:
            ds = load_dataset(dataset_name)
    except ValueError as e:
        logging.warning(e)
        config = input("Please choose from available configs: ")
        return prepare_dataset(dataset_name, config, dataset_to_use, prompt_key, answer_key)
        
    ds_keys = list(ds.keys())
    prompt_keys = {}
    collective_dict = {}
    for key in ds_keys:
        feature_cols = list(ds[key].features.keys())
        first_row_data = ds[key][0]
        adj_f_row = {}
        keys_per_item = []
        for f in feature_cols:
            keys_per_item.append(f)
            adj_f_row[f] = re.sub(r' +', ' ', first_row_data[f].replace('\n', ' ').replace('\t', ' ').replace('\r', ' '))[:100]
        prompt_keys[key] = keys_per_item
        result_dict = {
            "feature_columns": feature_cols,
            "row_count": len(ds[key]),
            "sample_row": adj_f_row
        }
        collective_dict[key] = result_dict
    print(json.dumps(collective_dict, indent=4))
    if len(ds_keys) == 1:
        dataset_to_use = ds_keys[0]
    else:
        while dataset_to_use.strip() not in ds_keys:
            print("Invalid Option! ", end="")
            logging.warning(f"Invalid Option {dataset_to_use}")
            dataset_to_use = input(f"Please choose a dataset to work fetch from {ds_keys}: ")
    prompt_keys = prompt_keys[dataset_to_use]
    if len(prompt_keys) == 1:
        prompt_key = prompt_keys[0]
    else:
        while prompt_key.strip() not in prompt_keys:
            print("Invalid Option! ", end="")
            logging.warning(f"Invalid or empty prompt_key: {prompt_key}")
            prompt_key = input(f"Please choose the prompt key to use: {list(prompt_keys)}: ")
    if len(prompt_keys) == 1:
        answer_key = None
    else:
        while answer_key.strip() not in prompt_keys:
            if answer_key == "":
                break
            print("Invalid Option! ", end="")
            logging.warning(f"Invalid or empty answer_key: {answer_key}")
            answer_key = input(f"Please choose the answer key to use: {list(prompt_keys)} [Leave empty if none]: ")
    return ds[dataset_to_use], prompt_key, answer_key

##############################################################################
# 3. TEXT CLEANING / PREPROCESSING
##############################################################################

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

##############################################################################
# 4. DISTANCE CALCULATIONS
##############################################################################

def compute_edit_distance(tokens_a, tokens_b):  # For text prompts | input tokens: list
    """
    Token-level edit distance (Levenshtein).
    """
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    dp = [[0]*(len_b+1) for _ in range(len_a+1)]

    for i in range(len_a+1):
        dp[i][0] = i
    for j in range(len_b+1):
        dp[0][j] = j

    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            if tokens_a[i-1] == tokens_b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    return dp[len_a][len_b]

def compute_edit_distance_levenshtein(str1, str2):  # For code prompts | input strings
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]
from chromadb.config import Settings
# ANONYMIZED_TELEMETRY=False
def initialize_chroma():
    chroma_client = chromadb.HttpClient(host=CHROMADB_IP, port=int(CHROMADB_PORT), settings=Settings(anonymized_telemetry=False))
    return chroma_client

def create_chroma_collection(collection_name, client):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=EMBEDDINGS_OPENAI_API_KEY,
        model_name="text-embedding-3-large"
    )
    created_collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=openai_ef, 
        metadata={"hnsw:space": "cosine"}
    )
    return created_collection

def delete_chroma_collection(collection_name, client):
    client.delete_collection(name=collection_name)

def answers_get_distance(chroma_collection, answer_to_check):
    distance = chroma_collection.query(query_texts=[answer_to_check], n_results=1)
    # num_of_q = chroma_collection.count()
    # chroma_collection.add(documents=[answer_to_check], ids=[f"id{num_of_q + 1}"])
    return distance['distances'][0][0]

def assign_similarity_threshold(greedy_answer, distance_type="edit", tokenizer=None):
    """
    Dynamically compute a 'reasonable' threshold based on:
      - The token length of 'greedy_answer' if distance_type='edit'.
      - A fixed threshold if distance_type='embeddings'.

    Args:
        greedy_answer (str): The reference text or greedy solution.
        distance_type (str): Either "edit" or "embeddings".

    Returns:
        float: 
          - For "edit": A fraction of the reference token length (used in cutoff).
          - For "embeddings": A fixed distance in [0,1] for cosine distance, etc.
    """
    tokens = tokenize_text(greedy_answer, tokenizer)
    length_ = len(tokens)

    if distance_type == "edit":
        # ---------------------------------------------------------
        # A length-based fraction for token-level edit distance
        # Example heuristic:
        #  - Short reference (< 30 tokens): allow 15% difference
        #  - Medium reference (30–99 tokens): allow 10% difference
        #  - Long reference (>=100 tokens): allow 5% difference
        # ---------------------------------------------------------
        if length_ < 30:
            return 0.15
        elif length_ < 100:
            return 0.1
        else:
            return 0.05
    
    elif distance_type == "embeddings":
        # -----------------------------------------------------------
        # Use a SINGLE fixed threshold for embedding-based distance
        # e.g., 0.14 means near-duplicates must have distance <= 0.14 
        # (cosine distance ~ 85% similarity).
        # -----------------------------------------------------------
        return 0.14
    
    else:
        raise ValueError("distance_type must be 'edit' or 'embeddings'")

##############################################################################
# 5. PEAKEDNESS COMPUTATION
##############################################################################

def compute_peakedness(outputs, threshold=0.05, mode="edit", greedy_0=False, tokenizer=None, edit_dp_or_lev="dp"):
    """
    If mode == "edit", uses token-level edit distance vs. a fraction-of-reference-len cutoff.
    If mode == "embeddings", uses an absolute distance in [0,1], e.g. cosine distance.
    """
    if len(outputs) < 2:
        return 0.0

    if mode == "edit":
        if edit_dp_or_lev == "dp":
            tokenized_outputs = [tokenize_text(o, tokenizer) for o in outputs]

        if greedy_0:
            range_ = range(len(outputs))
            denom = len(outputs)
        else:
            greedy_0 = outputs[0]
            range_ = range(1, len(outputs))
            denom = len(outputs) - 1
        if edit_dp_or_lev == "dp":
            greedy_tokens = tokenize_text(greedy_0, tokenizer)
            len_greedy = len(greedy_tokens)
        else:
            len_greedy = len(greedy_0)
        near_count = 0
        edit_distances = 0
        for i in range_:
            if edit_dp_or_lev == "dp":
                ed = compute_edit_distance(greedy_tokens, tokenized_outputs[i])
            else:
                ed = compute_edit_distance_levenshtein(greedy_0, outputs[i])
            edit_distances += ed
            cutoff = threshold * len_greedy
            if ed <= cutoff:
                near_count += 1
            print(ed, cutoff)

        peakedness_value = near_count / denom
        similarity_avg = edit_distances / denom

        return peakedness_value, similarity_avg

    elif mode == "embeddings":
        chroma_client = initialize_chroma()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-4]
        collection_name = f"chroma_col_{current_time}"
        collection_ = create_chroma_collection(collection_name, chroma_client)
        if greedy_0:
            greedy_answer = greedy_0
            range_ = range(len(outputs))
            denom = len(outputs)
        else:
            greedy_answer = outputs[0]
            range_ = range(1, len(outputs))
            denom = len(outputs) - 1

        collection_.add(documents=[greedy_answer], ids=[f"id1"])
        near_count = 0
        embeds_distances = 0
        for i in range_:
            dist = answers_get_distance(collection_, outputs[i])
            embeds_distances += dist
            if dist <= threshold:
                near_count += 1

        delete_chroma_collection(collection_name, chroma_client)
        
        peakedness_value = near_count / denom
        similarity_avg = embeds_distances / denom
        
        return peakedness_value, similarity_avg

    else:
        raise ValueError("Invalid mode. Use 'edit' or 'embeddings'.")

##############################################################################
# 6. TOKENIZERS
##############################################################################

def generate_tokenizer(model_name, tokenizer_mode=None):
    if tokenizer_mode == 0:
        # For example: "Salesforce/codegen-6B-multi", "codellama/CodeLlama-7b-hf", etc.
        return AutoTokenizer.from_pretrained(model_name)
    elif tokenizer_mode == 1:
        # For example: "gpt-3.5-turbo", "gpt-4", etc.
        return tiktoken.encoding_for_model(model_name)
    else:
        return None

def tokenize_text(text, tokenizer=None):
    if tokenizer:
        return tokenizer.encode(text)
    return nltk.word_tokenize(text)