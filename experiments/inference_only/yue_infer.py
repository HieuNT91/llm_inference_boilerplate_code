import pandas as pd 
from prompts import qwen_math_instruct_prompt_template
from transformers_utils import InferenceTransformers

model_repo = "Qwen/Qwen2.5-Math-1.5B-Instruct"
model_str = model_repo.split("/")[-1].lower().replace("-", "_")
prompt_template = qwen_math_instruct_prompt_template
test_df = pd.read_excel("datasets/YueData/yue_test.xlsx")
train_df = pd.read_excel("datasets/YueData/yue_train.xlsx")
df = pd.concat([train_df, test_df], ignore_index=True)
text_list = df['Question'].tolist()
text_list = [prompt_template.format(question=text) for text in text_list]

inference_transformers = InferenceTransformers(model_repo, use_auto_model=True)
sorted_text_list, sorted_indices = inference_transformers.sort_inputs(text_list)


num_return_sequences = inference_transformers.config['num_return_sequences']  
max_new_tokens = inference_transformers.config['max_new_tokens']  
outputs = inference_transformers.generate_non_distributed(sorted_text_list, cache_path=f"tmp/yue_best_of_{num_return_sequences}_{max_new_tokens}_{model_str}/")
grouped_outputs = [
    outputs[i : i + num_return_sequences] for i in range(0, len(outputs), num_return_sequences)
]
reverse_indices = [0] * len(sorted_indices)
for pos, original_idx in enumerate(sorted_indices):
    reverse_indices[original_idx] = pos
ordered_groups = [grouped_outputs[reverse_indices[i]] for i in range(len(df))]
for i in range(num_return_sequences):
    df[f"answer {i+1}"] = [group[i] for group in ordered_groups]
output_file = f"tmp/yue_best_of_{num_return_sequences}_{max_new_tokens}_{model_str}/yue_result.csv"
df.to_csv(output_file, index=False)
print(f"Extended CSV file saved to: {output_file}")
    