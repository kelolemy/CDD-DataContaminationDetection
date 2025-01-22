import json
import re
import nltk
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from config import EMBEDDINGS_OPENAI_API_KEY, CHROMADB_IP, CHROMADB_PORT


##############################################################################
# 1. API CALLS
##############################################################################

from LLMS_APIS.gpt4o import call_gpt4o as call_api  # REPLACE WITH THE LLM TO TEST

def call_api_wrapper(prompt, temperature=0.7, n=1):
    """
    Wraps the call_api function to request multiple outputs concurrently.
    Adjust max_workers as you see fit.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_api, prompt, temperature) for _ in range(n)]
        outputs = [f.result() for f in futures]
    return outputs

def call_llm_get_greedy(prompt):
    greedy_answer = call_api(prompt, temperature=0.0)
    return greedy_answer

##############################################################################
# 2. LOAD DATASET
##############################################################################

def prepare_dataset(dataset_name, config_split=None, dataset_to_use=None, prompt_key=None, answer_key=None):
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
        print(e)
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
            dataset_to_use = input(f"Please choose a dataset to work fetch from {ds_keys}: ")
    prompt_keys = prompt_keys[dataset_to_use]
    if len(prompt_keys) == 1:
        prompt_key = prompt_keys[0]
    else:
        while prompt_key.strip() not in prompt_keys:
            print("Invalid Option! ", end="")
            prompt_key = input(f"Please choose the prompt key to use: {list(prompt_keys)}: ")
    if len(prompt_keys) == 1:
        answer_key = None
    else:
        while answer_key.strip() not in prompt_keys:
            if answer_key == "":
                break
            print("Invalid Option! ", end="")
            answer_key = input(f"Please choose the answer key to use: {list(prompt_keys)} [Leave empty if none]: ")
    return ds[dataset_to_use], prompt_key, answer_key

##############################################################################
# 2. DISTANCE CALCULATIONS
##############################################################################

def compute_edit_distance(tokens_a, tokens_b):
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

def initialize_chroma():
    chroma_client = chromadb.HttpClient(host=CHROMADB_IP, port=int(CHROMADB_PORT))
    return chroma_client

def create_chroma_collection(collection_name, client):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=EMBEDDINGS_OPENAI_API_KEY,
            model_name="text-embedding-3-large"
        )
    created_collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
    return created_collection

def answers_get_distance(chroma_collection, answer_to_check):
    distance = chroma_collection.query(query_texts=[answer_to_check], n_results=1)
    # num_of_q = chroma_collection.count()
    # chroma_collection.add(documents=[answer_to_check], ids=[f"id{num_of_q + 1}"])
    distance = distance['distances'][0][0]
    return distance

def assign_threshold(greedy_answer, distance_type="edit"):
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
    tokens = nltk.word_tokenize(greedy_answer)
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
        # e.g., 0.15 means near-duplicates must have distance <= 0.15 
        # (cosine distance ~ 85% similarity).
        # -----------------------------------------------------------
        return 0.15
    
    else:
        raise ValueError("distance_type must be 'edit' or 'embeddings'")


##############################################################################
# 3. PEAKEDNESS COMPUTATION
##############################################################################

def compute_peakedness(outputs, threshold=0.05, mode="edit", greedy_0=False):
    """
    Compute 'peakedness' of the distribution of LLM outputs.
    - If mode == "edit", uses token-level edit distance to measure near-duplicates.
    - If mode == "embeddings", uses embedding-based distance to measure similarity.

    Args:
        outputs (list of str): LLM-generated outputs.
        threshold (float): For "edit" mode, fraction of reference length used as a cutoff.
                           For "embeddings" mode, we can interpret it as a numeric
                           distance threshold (lower = more similar).
        mode (str): "edit" or "embeddings".

    Returns:
        float: A number in [0,1] indicating fraction of near-duplicates among the samples.
               Higher => more suspicious / peaked / possibly contaminated.
    """
    if len(outputs) < 2:
        return 0.0

    # Ephemeral "collection" approach (especially for embeddings) 
    # to store intermediate representations or check similarity.
    # We remove or discard it at the end.

    if mode == "edit":
        # ------------------------------------------------
        # Peakedness via token-level edit distance
        # We'll define "greedy" as outputs[0]
        # Or greedy_0 if given
        # ------------------------------------------------
        tokenized_outputs = [nltk.word_tokenize(o) for o in outputs]
        if greedy_0:
            greedy_tokens = nltk.word_tokenize(greedy_0)
            range_ = range(len(outputs))
            len_ = len(outputs)
        else:
            greedy_tokens = tokenized_outputs[0]
            range_ = range(1, len(outputs))
            len_ = len(outputs) - 1
        len_greedy = len(greedy_tokens)

        near_count = 0
        for i in range_:
            ed = compute_edit_distance(greedy_tokens, tokenized_outputs[i])
            # cutoff = threshold * length_of_greedy
            cutoff = threshold * len_greedy
            if ed <= cutoff:
                near_count += 1

        peakedness_value = near_count / len_
        return peakedness_value

    elif mode == "embeddings":
        # ------------------------------------------------
        # Peakedness via embedding-based distance
        # We'll define "greedy" as outputs[0]
        # Or greedy_0 if given
        # ------------------------------------------------
        chroma_client = initialize_chroma()
        collection_name = f"chroma_collection_{str(datetime.now())}"
        collection_ = create_chroma_collection(collection_name, chroma_client)
        if greedy_0:
            greedy_answer = greedy_0
            range_ = range(len(outputs))
            len_ = len(outputs)
        else:
            greedy_answer = outputs[0]
            range_ = range(1, len(outputs))
            len_ = len(outputs) - 1

        collection_.add(documents=[greedy_answer], ids=[f"id1"])
        near_count = 0

        for i in range_:
            dist = answers_get_distance(collection_, outputs[i])
            # If distance < threshold => near-duplicate in embedding space
            if dist <= threshold:
                near_count += 1

        peakedness_value = near_count / len_
        return peakedness_value

    else:
        raise ValueError("Invalid mode. Use 'edit' or 'embeddings'.")
