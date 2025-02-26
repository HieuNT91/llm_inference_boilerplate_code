from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List, Dict
import json
from pathlib import Path
import torch 
import transformers 
from functools import partial
import numpy as np
from peft import PeftModel
from accelerate.utils import gather_object
import pickle
import os 
from hooker import BaseHooker, ZeroOutHooker
from tqdm import tqdm 

class InferenceTransformers:
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='causal_lm',
                 use_auto_model: bool = True,
                 use_accelerate: bool = False):
        
        self.model_repo = model_repo
        self.is_distributed_generation = use_accelerate
        
        self.model_name = model_repo.split('/')[-1]
        self.default_generation_config_path = "config/default_generation_config.json"
        if config is not None:
            self.config = config
        else:
            self.config = self.load_config(self.default_generation_config_path)
        
        if self.is_distributed_generation:
            from accelerate import PartialState, Accelerator, InitProcessGroupKwargs
            from datetime import timedelta
            import torch.distributed as dist
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.distributed_state = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))])
        
        self.init_model_tokenizer(model_repo, 
                                lora_path=lora_path,
                                use_auto_model=use_auto_model,
                                task_type=task_type,
                                attention_implementation=self.config.get("attention_implementation", 'eager'),
                                use_accelerate=use_accelerate)
            
    def init_model_tokenizer(self,
                             model_repo: str,
                             lora_path: str = None, 
                             use_auto_model: bool = True,
                             task_type: str = 'causal_lm',
                             attention_implementation: str = 'eager',
                             use_accelerate: bool = False):
        
        if task_type not in {'seq_cls', 'causal_lm'}:
            raise ValueError(f"Task type {task_type} not supported.")
        if attention_implementation not in {'eager', 'flash_attention_2'}:
            raise ValueError(f"Attention implementation {attention_implementation} not supported.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        if use_auto_model:
            if task_type == 'causal_lm':
                self.model = AutoModelForCausalLM.from_pretrained(model_repo if lora_path is None else lora_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                            attn_implementation=attention_implementation)
                
            elif task_type == 'seq_cls':
                self.model = AutoModelForSequenceClassification.from_pretrained(model_repo if lora_path is None else lora_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                            attn_implementation=attention_implementation)
        else:
            if 'qwen' in model_repo.lower():
                from mod_llm.qwen2.modeling_qwen2 import Qwen2ForCausalLM
                self.model = Qwen2ForCausalLM.from_pretrained(model_repo, 
                                                        torch_dtype=torch.float16, 
                                                        device_map='auto' if not use_accelerate else self.distributed_state.device,
                                                        attn_implementation=attention_implementation)
                if lora_path is not None:
                    # self.add_special_tokens()
                    self.model = PeftModel.from_pretrained(
                                self.model,
                                lora_path,
                                is_trainable=False,
                            )
                
            # elif 'gemma' in model_repo.lower():
            #     from .mod_llm.gemma2.modeling_gemma2 import GemmaForCausalLM
            #     self.model = GemmaForCausalLM.from_pretrained(model_repo, 
            #                                             torch_dtype=torch.float16, 
            #                                             device_map='auto' if not use_accelerate else self.distributed_state.device,
            #                                             attn_implementation=attention_implementation)
            else:
                raise ValueError(f"Model {model_repo} not supported. or set use_auto_model=True instead")
        
        print(f'Model loaded from {model_repo if lora_path is None else lora_path}')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model.eval()
        self.device = self.model.device if not use_accelerate else self.distributed_state.device
    
    
    @staticmethod
    def save_config(config: Dict, filepath: str = "config/default_generation_config.json"):
        """Save generation configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod    
    def load_config(filepath: str = "src/config/default_generation_config.json") -> Dict:
        """Load generation configuration from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Config file not found at {filepath}.")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def prepare_inputs_for_generation(self, inputs):
        """
        Processes and checks the validity of inputs for text generation.

        Args:
            inputs (str, list of str, or dict): The input(s) to process for generation.
                - str: A single string input.
                - list of str: A list of input strings. (only accept this when using distributed generation)
                - dict: A preprocessed input dictionary compatible with the model.

        Returns:
            model_inputs: The processed inputs, ready for the model (e.g., tensors or BatchEncoding).

        Raises:
            ValueError: If the input type is unsupported.
        """
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            # Tokenize a list of strings with padding
            # model_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
            batch_size_per_device = self.config.get('batch_size_per_device')
            cache_interval = self.config.get('cache_interval')
            formatted_prompts = [inputs[i : i + batch_size_per_device] for i in range(0, len(inputs), batch_size_per_device)]
            model_inputs =[self.tokenizer(formatted_prompt, padding=True, return_tensors='pt')
                            for formatted_prompt in formatted_prompts]
            cache_interval_ = cache_interval if not self.is_distributed_generation else cache_interval * self.distributed_state.num_processes
            split_model_inputs = [model_inputs[i : i + cache_interval_] for i in range(0, len(model_inputs), cache_interval_)]
            return split_model_inputs
        else:
            if self.is_distributed_generation:
                raise ValueError(f"Only accept list of str for distributed generation.")
        
        if isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            # If inputs are already a dictionary or BatchEncoding, use them directly
            model_inputs = inputs
        elif isinstance(inputs, str):
            # Tokenize a single string
            model_inputs = self.tokenizer(inputs, return_tensors="pt")
        else:
            # Raise an error for unsupported input types
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")

        return model_inputs

    
    def prepare_config_for_generation(self, **kwargs):
        """
        Prepares and returns a configuration dictionary for text generation.
        Args:
            **kwargs: Required keyword arguments to customize generation settings.
        Returns:
            dict: A dictionary containing the generation configuration.
        Raises:
            KeyError: If a required parameter is missing in kwargs.
        """
        # TODO: Modify this as you changed the config
        required_keys = [ 
            "temperature",
            "max_new_tokens",
            "top_k",
            "top_p",
            "repetition_penalty",
            "do_sample",
            "use_cache",
            "num_return_sequences",
        ]

        for key in required_keys:
            if key not in kwargs:
                raise KeyError(f"Missing required generation parameter: '{key}'")
        
        generation_config = {key: kwargs[key] for key in required_keys}
        generation_config["pad_token_id"] = kwargs.get("pad_token_id", self.tokenizer.pad_token_id)
        generation_config["bos_token_id"] = kwargs.get("bos_token_id", self.tokenizer.bos_token_id)
        generation_config["eos_token_id"] = kwargs.get("eos_token_id", self.tokenizer.eos_token_id)
        return generation_config
    
    def count_token(self, input_ids): # this function exclude all special tokens
        special_token_ids = set(self.tokenizer.all_special_ids) 
        token_lengths = []
        for sample in input_ids:
            length = sum(1 for token_id in sample.tolist() if token_id not in special_token_ids)
            token_lengths.append(length)
        return token_lengths
    
    def cache_inputs_outputs(self, inputs, outputs, input_token_lengths, output_token_lengths, group_tag, cache_path):
        os.makedirs(cache_path, exist_ok=True)
        file_path = os.path.join(cache_path, f"{group_tag}_cache.json")
        num_return_sequences = self.config.get("num_return_sequences", 1)
        grouped_outputs = [
            outputs[i : i + num_return_sequences] for i in range(0, len(outputs), num_return_sequences)
        ]
        grouped_output_token_lengths = [
            output_token_lengths[i : i + num_return_sequences] for i in range(0, len(output_token_lengths), num_return_sequences)
        ]
        group_data = [
            {"input": inp, 
             "output": out, 
             "input_token_length": inp_length,
             "output_token_lengths": out_length} for inp, out, inp_length, out_length in zip(inputs, grouped_outputs, input_token_lengths, grouped_output_token_lengths)
        ]
        with open(file_path, "w") as file:
            json.dump(group_data, file, indent=4)

        print(f"Cached inputs and outputs for group '{group_tag}' saved to {cache_path}.")
        
    
    @staticmethod
    def load_cached_outputs(cache_path, group_tag):
        file_path = os.path.join(cache_path, f"{group_tag}_cache.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                cached_data = json.load(file)
            # Extract only the outputs from the cached data
            return [entry["output"] for entry in cached_data]
        return None
    
    def sort_inputs(self, inputs: List[str]):
        token_lengths = [len(self.tokenizer(text)["input_ids"]) for text in inputs]
        indexed_inputs = list(enumerate(inputs))
        # Sort by token length, keeping track of original indices
        sorted_indexed_inputs = sorted(zip(indexed_inputs, token_lengths), key=lambda x: x[1])
        sorted_indices = [item[0][0] for item in sorted_indexed_inputs]
        sorted_inputs = [item[0][1] for item in sorted_indexed_inputs]
        return sorted_inputs, sorted_indices
    
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        cache_path: str,
    ) -> List[str]:
        if self.is_distributed_generation:
            raise NotImplementedError("Distributed generation not supported yet.")
            return self.generate_distributed(inputs, cache_path)
        else:
            return self.generate_non_distributed(inputs, cache_path)
        
    def generate_non_distributed(
        self,
        inputs: Union[str, List[str], Dict],
        cache_path: str,
    ) -> List[str]:
        """
        Generates model outputs for the given inputs, using caching to skip regeneration
        for already-processed inputs.

        Args:
            cache_path (str): Path to the directory where cached outputs are stored.
            inputs (Union[str, List[str], Dict]): Input text(s) or tokenized inputs for generation.

        Returns:
            List[str]: Generated outputs (decoded).
        """
        # Prepare generation configuration and inputs
        generation_config = self.prepare_config_for_generation(**self.config)
        model_inputs = self.prepare_inputs_for_generation(inputs)

        # Handle single input (not a list) case
        if not isinstance(model_inputs, list):
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            seq_len = model_inputs['input_ids'].shape[1]
            output = self.model.generate(
                **model_inputs,
                **generation_config,
            )
            return self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True)

        # Handle grouped inputs (list of batches)
        outputs = []
        for group_tag, input_batches in enumerate(model_inputs):
            group_tag_str = f"group_{group_tag}"  # Unique cache identifier for the group

            # Check if outputs for this group are already cached
            cached_outputs = self.load_cached_outputs(cache_path, group_tag_str)
            if cached_outputs is not None:
                print(f"Loaded cached outputs for {group_tag_str}.")
                outputs.extend([cached_output for sublist in cached_outputs for cached_output in sublist])
                continue  # Skip generation for this group

            group_outputs = []
            inputs_to_cache = []
            input_token_lengths = []
            output_token_lengths = []
            for batch in input_batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                seq_len = batch['input_ids'].shape[1]
                output = self.model.generate(
                    **batch,
                    **generation_config,
                )
                input_token_lengths.extend(self.count_token(batch['input_ids']))
                output_token_lengths.extend(self.count_token(output[:, seq_len:]))
                inputs_to_cache.extend(self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
                group_outputs.extend(self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True))

            outputs.extend(group_outputs)
            self.cache_inputs_outputs(inputs=inputs_to_cache, 
                                      outputs=group_outputs, 
                                      input_token_lengths=input_token_lengths,
                                      output_token_lengths=output_token_lengths,
                                      group_tag=group_tag_str, 
                                      cache_path=cache_path)
            
        return outputs
    
    # def generate_distributed(
    #     self,
    #     inputs: Union[str, List[str], Dict],
    #     return_raw_output: bool = False,
    # ):
        
    #     generation_config = self.prepare_config_for_generation(**self.config)
    #     model_inputs = self.prepare_inputs_for_generation(inputs)
    #     seq_len = model_inputs['input_ids'].shape[1]
        
    #     model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}
    #     output = self.model.generate(
    #         **model_inputs,
    #         **generation_config,
    #     )
    #     if return_raw_output:
    #         return output
    #     else:
    #         response = self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True)
    #         return response
    
    def forward(
        self,
        inputs: Union[str, List[str], Dict],
    ) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        
        model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}
        output = self.model(**model_inputs, use_cache=False)
        
        return output


if __name__ == '__main__':
    from prompt_utils import load_all_prompts, get_prompt
    import pandas as pd 
    model_repo = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # all_prompts = load_all_prompts("src/prompts")
    # prompt_template = get_prompt(all_prompts, "yue", "qwen_math", "vanilla_instruct")['prompt']
    prompt_template = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    test_df = pd.read_excel("datasets/YueData/yue_test.xlsx")
    train_df = pd.read_excel("datasets/YueData/yue_train.xlsx")
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    text_list = df['Question'].tolist()
    text_list = [prompt_template.format(question=text) for text in text_list]
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    inference_transformers = InferenceTransformers(model_repo, use_auto_model=True)
    sorted_text_list, sorted_indices = inference_transformers.sort_inputs(text_list)

    
    num_return_sequences = inference_transformers.config['num_return_sequences']  
    max_new_tokens = inference_transformers.config['max_new_tokens']  
    outputs = inference_transformers.generate_non_distributed(sorted_text_list, cache_path=f"tmp/yue_best_of_{num_return_sequences}_{max_new_tokens}_1.5B_/")
    grouped_outputs = [
        outputs[i : i + num_return_sequences] for i in range(0, len(outputs), num_return_sequences)
    ]
    reverse_indices = [0] * len(sorted_indices)
    for pos, original_idx in enumerate(sorted_indices):
        reverse_indices[original_idx] = pos
    ordered_groups = [grouped_outputs[reverse_indices[i]] for i in range(len(df))]
    for i in range(num_return_sequences):
        df[f"answer {i+1}"] = [group[i] for group in ordered_groups]
    output_file = f"tmp/yue_best_of_{num_return_sequences}_{max_new_tokens}/yue_extended_1.5B.csv"
    df.to_csv(output_file, index=False)
    print(f"Extended CSV file saved to: {output_file}")
    