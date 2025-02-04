from src.main import main


def caller():
    # -----------------------------------------------------------------------------------
    # Fill in below: || LOOK AT `example_configurations` for examples for each mode
    # -----------------------------------------------------------------------------------
    DATA_INPUT_MODE = "datasets"        # SET TO "datasets", "strings", OR "jsons"
    
    # IF DATA_INPUT_MODE = "datasets"; SET:
    D_DATASET_NAME        = "gsm8k"
    D_CONFIG_SPLIT        = "main"      # Optional
    D_SUBDATASET_TO_USE   = "test"      # Optional | You'll be prompted to give if there's more than one subdatasets.
    D_PROMPT_KEY          = "question"  # Optional | You'll be prompted to give if there's more than one key.
    D_ANSWER_KEY          = "answer"    # Optional
    
    # IF DATA_INPUT_MODE = "jsons"; SET:
    J_BENCHMARK_LOC       = ""          # Set to location of benchmark json
    J_PROMPT_KEY          = ""          # Prompt key in the specified json.
    J_ANSWER_KEY          = ""          # Answer key in the specified json.
    
    # IF DATA_INPUT_MODE = "strings"; SET:
    S_BENCHMARK_LOC       = ""          # Set to location of benchmark json
    
    # CONFIG OPTIONS
    MODEL_NAME          = "gpt-4o"      # Model in test
    DISTANCE_MODE       = "edit"        # or "embeddings"
    SIM_THRESHOLD       = 0.05          # fraction of token/embeddings similarity cutoff
    TEMPERATURE         = 0.7           # sampling temperature for LLM
    NUM_SAMPLES         = 5             # how many outputs to sample (5 - 20)
    CONTAM_THRESH       = 0.2           # if peakedness > this => "contaminated"
    NUMBER_OF_ENTRIES   = 100             # The number of entries to benchmark the model.
    GREEDY_0_TEMP       = False         # Gets 0.0 temperature response from LLM, use it as reference
    ENFORCE_GREEDY      = False         # Forces to use GREEDY_0_TEMP answer, even when there's answer provided in benchmark.
    BENCHMARK_CODE      = False         # "code", "text", "none": skip cleaning
    TOKENIZER_MODE      = 0             # 0: transformers.AutoTokenizer, 1: tiktoken, 2: nltk
    TOKENIZER_NAME      = ""            # if TOKENIZER_MODE == 0 | 1, give model_name, else None
    EDIT_DP_OR_LEV      = "dp"          # For DISTANCE_MODE == "edit"; 
                                        #   "dp": dynamic programming edit distance, "lev": levenshtein edit distance,
                                        # Set to None, if DISTANCE_MODE == "embeddings".
    # -----------------------------------------------------------------------------------
    
    
    # ----------------------------------- Don't Touch -----------------------------------
    input_mode = DATA_INPUT_MODE
    if DATA_INPUT_MODE == "strings":
        input_conf = {"BENCHMARK_LOC": S_BENCHMARK_LOC}
    elif DATA_INPUT_MODE == "jsons":
        input_conf = {
            "BENCHMARK_LOC": J_BENCHMARK_LOC,
            "PROMPT_KEY": J_PROMPT_KEY,
            "ANSWER_KEY": J_ANSWER_KEY
        }
    elif DATA_INPUT_MODE == "datasets":
        input_conf = {
            "DATASET_NAME": D_DATASET_NAME,
            "CONFIG_SPLIT": D_CONFIG_SPLIT,
            "SUBDATASET_TO_USE": D_SUBDATASET_TO_USE,
            "PROMPT_KEY": D_PROMPT_KEY,
            "ANSWER_KEY": D_ANSWER_KEY
        }
    else:
        raise ValueError('DATA_INPUT_MODE must be "strings", "jsons", or "datasets"')

    main_conf = {
        "MODEL_NAME": MODEL_NAME,
        "DISTANCE_MODE": DISTANCE_MODE,
        "SIM_THRESHOLD": SIM_THRESHOLD,
        "TEMPERATURE": TEMPERATURE,
        "NUM_SAMPLES": NUM_SAMPLES,
        "CONTAM_THRESH": CONTAM_THRESH,
        "GREEDY_0_TEMP": GREEDY_0_TEMP,
        "NUMBER_OF_ENTRIES": NUMBER_OF_ENTRIES,
        "ENFORCE_GREEDY": ENFORCE_GREEDY,
        "BENCHMARK_CODE": BENCHMARK_CODE,
        "TOKENIZER_MODE": TOKENIZER_MODE,
        "TOKENIZER_NAME": TOKENIZER_NAME,
        "EDIT_DP_OR_LEV": EDIT_DP_OR_LEV
    }
    # -------------------------------- Don't Touch - END --------------------------------
    
    main(input_mode, input_conf, main_conf)


caller()
