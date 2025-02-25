import os
import yaml

model_repo_to_name = {
    "meta-llama/Meta-Llama-3-8B": "llama3",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "Qwen/Qwen2.5-Math-7B-Instruct": "qwen_math",
    "Qwen/Qwen2.5-Math-7B": "qwen_math",
}
    
def load_all_prompts(base_dir="prompts"):
    """
    Recursively load all YAML prompts from the base_dir directory.
    Returns a nested dictionary of form: 
    prompts_dict[task_name][model_name][prompt_name] = { ...prompt metadata... }
    """
    prompts_dict = {}

    # Loop over each folder at the top level (these are your tasks, e.g., 'summarization')
    for task_dir_name in os.listdir(base_dir):
        task_dir_path = os.path.join(base_dir, task_dir_name)

        # Skip if it's not a directory
        if not os.path.isdir(task_dir_path):
            continue

        # Create an entry in prompts_dict for this task
        prompts_dict[task_dir_name] = {}

        # Inside each task folder, read each YAML file (e.g., llama3.yaml, qwen.yaml)
        for filename in os.listdir(task_dir_path):
            if filename.endswith(".yaml"):
                model_name = os.path.splitext(filename)[0]  # e.g., "llama3" from "qwen.yaml"
                file_path = os.path.join(task_dir_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    prompts_list = yaml.safe_load(f)

                # If your YAML file contains a list of prompts
                # create a dict keyed by the prompt's "name"
                model_prompts = {}
                for prompt_item in prompts_list:
                    prompt_name = prompt_item["name"]
                    model_prompts[prompt_name] = prompt_item

                prompts_dict[task_dir_name][model_name] = model_prompts

    return prompts_dict

def get_prompt(all_prompts, task_name, model_name, prompt_name):
    return all_prompts[task_name][model_name][prompt_name]

if __name__ == "__main__":
    all_prompts = load_all_prompts("src/prompts")
    prompt = get_prompt(all_prompts, "math_500", "qwen_math", "vanilla_instruct")
    breakpoint()
    print(prompt)