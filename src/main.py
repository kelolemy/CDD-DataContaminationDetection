import json
from dotenv import load_dotenv
from src.utils import call_api_wrapper, call_llm_get_greedy, compute_peakedness, prepare_dataset, assign_threshold, datetime

def main(input_mode, input_conf, main_conf):
    # 1. Load environment variables (optional, if needed)
    load_dotenv()
    
    # import nltk # only first time
    # nltk.download('punkt')
    
    # ----------------------------------------------------------------------
    # CONFIGURATION:
    # ----------------------------------------------------------------------
    DATA_INPUT_MODE = input_mode

    DATASET_NAME = input_conf.get('DATASET_NAME')
    CONFIG_SPLIT = input_conf.get('CONFIG_SPLIT')
    SUBDATASET_TO_USE = input_conf.get('SUBDATASET_TO_USE')
    PROMPT_KEY = input_conf.get('PROMPT_KEY')
    ANSWER_KEY = input_conf.get('ANSWER_KEY')
    BENCHMARK_LOC = input_conf.get('BENCHMARK_LOC')

    MODEL_NAME = main_conf.get('MODEL_NAME')
    DISTANCE_MODE = main_conf.get('DISTANCE_MODE')
    SIM_THRESHOLD = main_conf.get('SIM_THRESHOLD')
    TEMPERATURE = main_conf.get('TEMPERATURE')
    NUM_SAMPLES = main_conf.get('NUM_SAMPLES')
    CONTAM_THRESH = main_conf.get('CONTAM_THRESH')
    GREEDY_0_TEMP = main_conf.get('GREEDY_0_TEMP')
    NUMBER_OF_ENTRIES = main_conf.get('NUMBER_OF_ENTRIES')
    ENFORCE_GREEDY = main_conf.get('ENFORCE_GREEDY')
    # ----------------------------------------------------------------------

    # 2. Load or define your benchmark data
    if DATA_INPUT_MODE == "datasets":
        data = prepare_dataset(DATASET_NAME, CONFIG_SPLIT, SUBDATASET_TO_USE, PROMPT_KEY, ANSWER_KEY)
        benchmark, PROMPT_KEY, ANSWER_KEY = data
    else:
        with open(BENCHMARK_LOC, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

    # 3. Detect Contamination
    contamination_results = []
    if NUMBER_OF_ENTRIES == 0:
        NUMBER_OF_ENTRIES = len(benchmark)
    for idx, item in enumerate(benchmark):
        if idx >= NUMBER_OF_ENTRIES:
            break
        if idx % 10 == 0:
            print(f"Processed {idx} entries / {NUMBER_OF_ENTRIES}.")
        # A. Extract the prompt (and answer (o)) depending on the data-input mode
        if DATA_INPUT_MODE == "strings":
            prompt = item
            greedy_output = None
        else:
            prompt = item[PROMPT_KEY]
            greedy_output = item.get(ANSWER_KEY, None)

        if (not greedy_output and GREEDY_0_TEMP) or ENFORCE_GREEDY:
            greedy_output = call_llm_get_greedy(prompt)
        
        CONTAM_THRESH = assign_threshold(greedy_output, DISTANCE_MODE) # FOR NOW! TO BE UPDATED.
        # B. Sample multiple outputs
        outputs = call_api_wrapper(prompt, temperature=TEMPERATURE, n=NUM_SAMPLES)
        
        # C. Compute peakedness (edit-distance or embeddings) 
        peakedness_value = compute_peakedness(
            outputs, 
            threshold=SIM_THRESHOLD, 
            mode=DISTANCE_MODE,
            greedy_0=greedy_output
        )
        
        # D. Decide if it's leaked or not
        is_contaminated = (peakedness_value > CONTAM_THRESH)
        
        contamination_results.append({
            "index": idx,
            "prompt": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "peakedness": peakedness_value,
            "is_contaminated": is_contaminated
        })

    # 4. Print or save results
    print("=== Contamination Results ===")
    # for r in contamination_results:
    #     print(f"Index: {r['index']}, Prompt: {r['prompt']}")
    #     print(f"  Peakedness: {r['peakedness']:.4f}, Contaminated: {r['is_contaminated']}")
    #     print("")

    fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace('-', '')
    RESULTS_F_LOC = f"benchmark_results/{MODEL_NAME}_{fname}.json"
    
    full_report = {
        "Input Mode": input_mode,
        "Input Configuration": input_conf,
        "Main Configuration": main_conf,
        "Contamination Results": contamination_results
    }
    # Save to a JSON file
    with open(RESULTS_F_LOC, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to {RESULTS_F_LOC}")


if __name__ == "__main__":
    main()
