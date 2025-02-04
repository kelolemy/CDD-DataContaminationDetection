import time
import nltk
import logging
import json
from dotenv import load_dotenv
from src.utils import (
    call_api_wrapper, 
    call_llm_get_greedy, 
    compute_peakedness, 
    prepare_dataset,
    assign_similarity_threshold,
    strip_and_truncate,
    generate_tokenizer,
    datetime
)

def main(input_mode, input_conf, main_conf):
    # 1. Load environment variables
    load_dotenv()

    # 2. Basic logging setup
    # logging.basicConfig(level=logging.INFO)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate the log filename
    log_filename = f"logs/log_{current_time}.log"

    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Optional: To print logs to console as well
        ]
    )
    
    class ExcludeHttpRequestsFilter(logging.Filter):
        def filter(self, record):
            # Exclude logs that contain "HTTP Request"
            # exclude_keys = ["HTTP Request", "Giving up send_request", "Backing off send_request"]
            exclude_keys = ["Nigga man that's bullshit"]
            exclude_cond = not bool([i for i in exclude_keys if i in record.getMessage()])
            return exclude_cond

    # Apply the filter to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(ExcludeHttpRequestsFilter())

    # Ensure the root logger handlers also apply the filter
    logging.getLogger().addFilter(ExcludeHttpRequestsFilter())

    # 3. Configuration extraction
    DATA_INPUT_MODE         = input_mode
    DATASET_NAME            = input_conf.get('DATASET_NAME')
    CONFIG_SPLIT            = input_conf.get('CONFIG_SPLIT')
    SUBDATASET_TO_USE       = input_conf.get('SUBDATASET_TO_USE')
    PROMPT_KEY              = input_conf.get('PROMPT_KEY')
    ANSWER_KEY              = input_conf.get('ANSWER_KEY')
    IMAGE_KEY              = input_conf.get('IMAGE_KEY')
    BENCHMARK_LOC           = input_conf.get('BENCHMARK_LOC')

    MODEL_NAME              = main_conf.get('MODEL_NAME')
    DISTANCE_MODE           = main_conf.get('DISTANCE_MODE')
    SIM_THRESHOLD           = main_conf.get('SIM_THRESHOLD')
    TEMPERATURE             = main_conf.get('TEMPERATURE')
    NUM_SAMPLES             = main_conf.get('NUM_SAMPLES')
    CONTAM_THRESH           = main_conf.get('CONTAM_THRESH')
    GREEDY_0_TEMP           = main_conf.get('GREEDY_0_TEMP')
    NUMBER_OF_ENTRIES       = main_conf.get('NUMBER_OF_ENTRIES')
    ENFORCE_GREEDY          = main_conf.get('ENFORCE_GREEDY')
    BENCHMARK_CODE          = main_conf.get('BENCHMARK_CODE', False)
    TOKENIZER_MODE          = main_conf.get('TOKENIZER_MODE', None)
    TOKENIZER_NAME          = main_conf.get('TOKENIZER_NAME', None)
    EDIT_DP_OR_LEV          = main_conf.get('EDIT_DP_OR_LEV')
    

    # 4. Print out pipeline configuration
    logging.info("=== Pipeline Configuration ===")
    for k, v in main_conf.items():
        logging.info(f"{k}: {v}")
    logging.info("=============================\n")

    # 5. Timer start
    start_time = time.time()
    
    # 6. Download/verify nltk tokenizers if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')

    # 7. Load benchmark data
    if DATA_INPUT_MODE == "datasets":
        ds_data = prepare_dataset(
            DATASET_NAME, 
            CONFIG_SPLIT, 
            SUBDATASET_TO_USE, 
            PROMPT_KEY, 
            ANSWER_KEY,
            IMAGE_KEY
        )
        benchmark, PROMPT_KEY, ANSWER_KEY, IMAGE_KEY = ds_data
    else:
        with open(BENCHMARK_LOC, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

    contamination_results = []
    outputs_json = []
    if NUMBER_OF_ENTRIES == 0:
        NUMBER_OF_ENTRIES = len(benchmark)
    elif NUMBER_OF_ENTRIES == -1:
        print(f"Found {len(benchmark)} entries in dataset. ", end="")
        while not (input_num := input("Enter number of entries: ").strip()).lstrip('-').isdigit():
            print("Invalid input. Please enter a valid integer.")
        NUMBER_OF_ENTRIES = int(input_num)

    # 8. Main loop
    for idx, item in enumerate(benchmark):
        if idx >= NUMBER_OF_ENTRIES:
            break

        if idx % 10 == 0:
            logging.info(f"Processed {idx} entries / {NUMBER_OF_ENTRIES} ...")

        # A. Extract prompt/answer
        if DATA_INPUT_MODE == "strings":
            prompt = strip_and_truncate(item, BENCHMARK_CODE)
            greedy_output = None
        else:
            prompt = strip_and_truncate(item[PROMPT_KEY], BENCHMARK_CODE)
            greedy_output = item.get(ANSWER_KEY, None)
            include_image = item.get(IMAGE_KEY)
        
        # B. Possibly get a greedy_0 output or ENFORCE_GREEDY
        if (not greedy_output and GREEDY_0_TEMP) or ENFORCE_GREEDY:
            greedy_output = call_llm_get_greedy(prompt, include_image)
        import re
        if len(greedy_output) < 5 or (len(greedy_output) == (len(re.sub(r'\D', '', greedy_output)) + greedy_output.count('.'))):
            continue
            
        # FOR NOW! TO BE UPDATED.
        tokenizer = generate_tokenizer(TOKENIZER_NAME, TOKENIZER_MODE)
        if SIM_THRESHOLD == 0:
            SIM_THRESHOLD = assign_similarity_threshold(greedy_output, DISTANCE_MODE, tokenizer)
        # C. Call LLM multiple times
        
        # ONLINE MODE START
        outputs = call_api_wrapper(prompt, include_image, temperature=TEMPERATURE, n=NUM_SAMPLES)
        # ONLINE MODE  END
        
        # D. Compute peakedness
        peakedness_value, similarity_distance = compute_peakedness(
            outputs, 
            threshold=SIM_THRESHOLD, 
            mode=DISTANCE_MODE,
            greedy_0=greedy_output,
            tokenizer=tokenizer,
            edit_dp_or_lev=EDIT_DP_OR_LEV
        )

        # E. Contamination decision
        is_contaminated = (peakedness_value > CONTAM_THRESH)
        
        if peakedness_value >= 0.7:
            case_ = "exact"
        elif peakedness_value >= CONTAM_THRESH:
            case_ = "exact + paraphrased"
        else:
            case_ = None
        
        if DISTANCE_MODE == "embeddings" and is_contaminated:
            processed_row = {
                "index": idx,
                # "task_id": item['id'],
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "similarity_distance": similarity_distance,
                "peakedness": peakedness_value,
                "is_contaminated": is_contaminated,
                "contamination_severity": case_
            }
        else:
            processed_row = {
                "index": idx,
                # "task_id": item['id'],
                "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "peakedness": peakedness_value,
                "is_contaminated": is_contaminated,
                "contamination_severity": case_
            }
        contamination_results.append(processed_row)
        processed_row['outputs'] = outputs
        processed_row['prompt'] = prompt
        processed_row['correct_answer'] = greedy_output
        # processed_row['answer_type'] = item['answer_type']
        processed_row['include_image'] = bool(include_image)
        outputs_json.append(processed_row)

    # 9. Summaries
    contaminated_count = sum(1 for r in contamination_results if r['is_contaminated'])
    total_count = len(contamination_results)
    contam_rate = 100.0 * contaminated_count / total_count if total_count else 0.0

    logging.info(f"Contaminated: {contaminated_count}/{total_count} ({contam_rate:.1f}%)")

    # 10. Timer end
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"Elapsed Time: {elapsed:.2f} seconds")

    # 11. Save final report
    fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace('-', '')
    RESULTS_F_LOC = f"benchmark_results/{MODEL_NAME}_{DISTANCE_MODE[:5]}_{f'{EDIT_DP_OR_LEV}_' if EDIT_DP_OR_LEV else ''}{fname}.json"
    
    full_report = {
        "Settings": {
            "Input Mode": input_mode,
            "Input Configuration": input_conf,
            "Main Configuration": main_conf,
        },
        "Contaminated Count": f"{contaminated_count}/{total_count} ({contam_rate:.1f}%)",
        "Contamination Results": contamination_results
    }

    with open(RESULTS_F_LOC, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    RESULTS_F_LOC = f"benchmark_results/answers/{MODEL_NAME}_{DISTANCE_MODE[:5]}_{f'{EDIT_DP_OR_LEV}_' if EDIT_DP_OR_LEV else ''}{fname}.json"
    with open(RESULTS_F_LOC, "w", encoding="utf-8") as f:
        json.dump(outputs_json, f, indent=2, ensure_ascii=False)
    logging.info(f"Report saved to {RESULTS_F_LOC}")


if __name__ == "__main__":
    main()
