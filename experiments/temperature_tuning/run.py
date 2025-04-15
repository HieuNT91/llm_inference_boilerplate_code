
from src.utils import load_jsonl
from src.inference_utils import InferenceEngine, CacheManager
import argparse
from src.prompts.qwen import MATH_PROMPT_TEMPLATE

def parse_args():
    """
    Parse command-line arguments for batch size and input range.
    
    Returns:
        argparse.Namespace: Parsed arguments with batch_size, input_start, and input_end.
    """
    parser = argparse.ArgumentParser(description="Process batch size and input range.")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Size of the batch to process (default: 16)"
    )
    parser.add_argument(
        "--input_start", 
        type=int, 
        default=0, 
        help="Start of the input range (default: 0)"
    )
    parser.add_argument(
        "--input_end", 
        type=int, 
        default=2500, 
        help="End of the input range (default: 2500)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = "data/math/test.jsonl"
    data = list(load_jsonl(data_path))[args.input_start: args.input_end]

    model_repo = "Qwen/Qwen2.5-Math-7B-Instruct"
    model_name = model_repo.split('/')[-1].lower()
    engine = InferenceEngine(model_repo, use_auto_model=True)


    def inference_fn(input_texts):
        response, topk_values, topk_indices, topk_tokens = engine.generate_with_topk_probs(
            inputs=input_texts,
            topk=5
        )
        return (response, topk_values, topk_indices, topk_tokens)

    dataset_name = 'math'
    cache_path = f"cache/{dataset_name}_{args.input_start}_{args.input_end}_{model_name}.pkl"
    prompts = [MATH_PROMPT_TEMPLATE.format(input=question['problem']) for question in data]

    cache_manager = CacheManager(inference_engine=engine, cache_file_path=cache_path, batch_size=args.batch_size)
    responses, topk_probs, topk_tokens = cache_manager.run_inference(prompts, topk=5, rerun=False)
    