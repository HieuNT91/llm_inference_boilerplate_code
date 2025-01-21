import json 
from tqdm import tqdm 
import pickle
import os 
from transformers_utils import IntervenableTransformers
from torch.utils.data import Dataset, DataLoader
from grading import grader 
import re 
from typing import Union, List, Dict
from collections import defaultdict
import numpy as np 
import logging
import matplotlib.pyplot as plt

def load_dataset(json_file):
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_dataset('/storage/hiu/project_2024/naacle/StudyNotes/datasets/MATH500/test.json')

model_name = 'Qwen/Qwen2.5-Math-7B-Instruct'
inference_transformers = IntervenableTransformers(model_name, use_auto_model=False)

default_config = IntervenableTransformers.load_config('config/default_generation_config.json')

def apply_prompt(sample, tokenizer):
    messages = [
    {"role": "system", "content": "Solve the given math problem."},
    {"role": "user", "content": f"{sample['Question']}.\n"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text 

def extract_boxed_answer(llm_output):
    start = llm_output.find(r'\boxed{')
    if start == -1:
        return None  # No \boxed{} found

    # Start parsing after \boxed{
    i = start + len(r'\boxed{')
    stack = ['{']  # Start with one opening brace
    content = []

    # Loop through the string to extract content inside \boxed{}
    while i < len(llm_output):
        char = llm_output[i]

        # Add characters to content
        content.append(char)

        # Handle braces to ensure proper matching
        if char == '{':
            stack.append('{')
        elif char == '}':
            stack.pop()
            # If the stack is empty, we've matched all braces
            if not stack:
                break

        i += 1

    # Join the content and remove the last closing brace
    result = ''.join(content).strip()
    return result[:-1].strip() if result.endswith('}') else result.strip()

class Math500Evaluator:
    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {'mean': [], 'var': []})
    
    def get_stats(self):
        print('average success rate:', np.mean(self.stats['success_rate']['mean']))
        print('average success rate variance:', np.mean(self.stats['success_rate']['var']))
        return self.stats
    
    def get_success_rate(self, model_output, ground_truth_answers, num_return_sequences, stat_track=False):
        assert len(model_output) % num_return_sequences == 0, "Output size must be divisible by num_return_sequences."
        assert len(ground_truth_answers) == len(model_output) // num_return_sequences, "Ground truth size mismatch with text_list."

        total_questions = len(ground_truth_answers)
        per_question_success_rates = []
        for i in range(total_questions):
            model_answers = model_output[i * num_return_sequences:(i + 1) * num_return_sequences]
            ground_truth_answer = ground_truth_answers[i]
            success_count = sum(
                grader.grade_answer(extract_boxed_answer(model_answer), ground_truth_answer) for model_answer in model_answers
            )
            per_question_success_rate = success_count / num_return_sequences
            per_question_success_rates.append(per_question_success_rate)

        if stat_track:
            self.stats['success_rate']['mean'].extend(per_question_success_rates)
            self.stats['success_rate']['var'].extend(per_question_success_rates)
            
        return per_question_success_rates, np.mean(per_question_success_rates), np.std(per_question_success_rates)


# Ensure the logging directory exists
log_dir = "../log"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, "experiment.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Visualization setup
fig, ax = plt.subplots()
ax.set_title("Success Rate vs. Pruned Layer")
ax.set_xlabel("Pruned Layer")
ax.set_ylabel("Mean Success Rate")
line, = ax.plot([], [], marker="o", label="Mean Success Rate")
shade = None
baseline_line = None  # Placeholder for the flat line
ax.legend()

# Parameters
batch_size = 7
text_list = []
ground_truth_answers = []
evaluator = Math500Evaluator()

# Assuming `inference_transformers` is an initialized model with a defined config
layers_to_prune = list(range(0, inference_transformers.model.config.num_hidden_layers, 2))

# Data placeholders for visualization
pruned_layers = []
mean_success_rates = []
std_devs = []

# Step 1: Evaluate no-prune baseline performance
logger.info("Evaluating no-prune baseline performance...")
baseline_success_rates = []
text_list = []
ground_truth_answers = []

for sample in data[:7]:  # Assuming `data` is your dataset
    text = apply_prompt(sample, inference_transformers.tokenizer)
    gt = sample['answer']
    text_list.append(text)
    ground_truth_answers.append(gt)

    if len(text_list) == batch_size:
        # Generate model outputs without pruning
        model_output = inference_transformers.generate(
            text_list,
            return_raw_output=False,
            heads_to_prune=[],  # No head pruning
            layers_to_prune=[],  # No layer pruning
            stat_track=False,
        )

        # Evaluate success rates
        per_question_success_rates, mean, std = evaluator.get_success_rate(
            model_output,
            ground_truth_answers,
            num_return_sequences=default_config.get("num_return_sequences"),
            stat_track=False,
        )
        baseline_success_rates.append(mean)
        text_list = []
        ground_truth_answers = []

# Calculate baseline mean and standard deviation
baseline_mean = np.mean(baseline_success_rates)
logger.info(f"Baseline Mean Success Rate: {baseline_mean}")

# Add a flat dotted line to the plot for baseline performance
baseline_line = ax.axhline(
    y=baseline_mean,
    color="red",
    linestyle="--",
    label="No-Prune Baseline",
)
ax.legend()

# Step 2: Main experiment loop for pruning
for layer in tqdm(layers_to_prune, desc="Pruning Layers"):
    logger.info(f"Pruning layer {layer}...")
    per_layer_success_rates = []
    text_list = []
    ground_truth_answers = []
    
    for sample in data[:8]:  # Assuming `data` is your dataset
        text = apply_prompt(sample, inference_transformers.tokenizer)
        gt = sample['answer']
        text_list.append(text)
        ground_truth_answers.append(gt)
        
        if len(text_list) == batch_size:
            # Generate model outputs with pruning
            model_output = inference_transformers.generate(
                text_list,
                return_raw_output=False,
                heads_to_prune=[2, 5],  # Assuming head pruning is skipped or modified
                layers_to_prune=[layer],
                stat_track=True if layer == 0 else False,
            )
            
            # Evaluate success rates
            per_question_success_rates, mean_success_rate, std_dev = evaluator.get_success_rate(
                model_output, 
                ground_truth_answers,
                num_return_sequences=default_config.get("num_return_sequences"), 
                stat_track=True,
            )
            logger.info(f"Layer {layer}: Mean={mean}, Std={std}")
            per_layer_success_rates.extend(per_question_success_rates)
            
            # Reset placeholders
            text_list = []
            ground_truth_answers = []
    
    # Log aggregated statistics for this layer
    mean_success_rate = np.mean(per_layer_success_rates)
    std_dev = np.std(per_layer_success_rates)
    logger.info(f"Layer {layer}: Aggregated Mean={mean_success_rate}, Std={std_dev}")
    
    # Store data for visualization
    pruned_layers.append(layer)
    mean_success_rates.append(mean_success_rate)
    std_devs.append(std_dev)
    
    # Update the plot dynamically
    line.set_xdata(pruned_layers)
    line.set_ydata(mean_success_rates)
    ax.relim()
    ax.autoscale_view()
    if shade:
        shade.remove()
    shade = ax.fill_between(
        pruned_layers, 
        np.array(mean_success_rates) - np.array(std_devs), 
        np.array(mean_success_rates) + np.array(std_devs), 
        alpha=0.2
    )
    plot_path = os.path.join(log_dir, f"success_rate_plot_layer_{layer}.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot for layer {layer} to {plot_path}")

# Ensure the final plot is saved
final_plot_path = os.path.join(log_dir, "success_rate_plot_final.png")
plt.savefig(final_plot_path)
logger.info(f"Saved final plot to {final_plot_path}")