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


class InferenceTransformers:
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='seq_cls',
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
                             task_type: str = 'seq_cls',
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
        required_keys = [ # TODO: Modify this as you changed the config
            "temperature",
            "max_new_tokens"
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
    
    def cache_inputs_outputs(self, inputs, outputs, group_tag, cache_path):
        
        os.makedirs(cache_path, exist_ok=True)
        file_path = os.path.join(cache_path, f"{group_tag}_cache.json")
        num_return_sequences = self.config.get("num_return_sequences", 1)
        grouped_outputs = [
            outputs[i : i + num_return_sequences] for i in range(0, len(outputs), num_return_sequences)
        ]
        group_data = [
            {"input": inp, "output": out} for inp, out in zip(inputs, grouped_outputs)
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
                outputs.extend(cached_outputs)
                continue  # Skip generation for this group

            group_outputs = []
            inputs_to_cache = []
            for batch in input_batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                seq_len = batch['input_ids'].shape[1]
                output = self.model.generate(
                    **batch,
                    **generation_config,
                )
                inputs_to_cache.extend(self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
                group_outputs.extend(self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True))

            outputs.extend(group_outputs)
            self.cache_inputs_outputs(inputs_to_cache, group_outputs, group_tag_str, cache_path)
            
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

    @torch.no_grad()
    def get_attention_at_layers(
        self,
        inputs: Union[str, List[str], Dict],
        layers_to_prune: List[int] = [3],
        stat_track: bool = True,
    ) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        attention_hooker = BaseHooker(layer_list=layers_to_prune, stat_track=stat_track)
        
        output = self.model(**model_inputs, 
                            edit_fn=attention_hooker,
                            use_cache=False)
        attention_hooker.log_stats()  
        print(f"Attention hooker has been called {attention_hooker.__call__.call_count} time(s).")
        return attention_hooker.attention


class IntervenableTransformers(InferenceTransformers):
    def __init__(self, model_repo: str, 
                 config: Dict = None, 
                 lora_path: str = None,
                 task_type='seq_cls',
                 use_auto_model: bool = True,
                 use_accelerate: bool = False):
        super().__init__(model_repo, 
                         config, 
                         lora_path, 
                         task_type, 
                         use_auto_model, 
                         use_accelerate=use_accelerate)
    
    def cache_outputs(self, outputs, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as out_f:
            pickle.dump(outputs, out_f)
            
    def load_cached_prompts(self, cache_path):
        with open(cache_path, "rb") as in_f:
            return pickle.load(in_f)
    
    def sort_inputs(
        self,
        inputs: List[str]
        ):
        token_lengths = [len(tokenizer(text)["input_ids"]) for text in inputs]
        inputs_with_lengths = zip(inputs, token_lengths)
        sorted_inputs_with_lengths = sorted(inputs_with_lengths, key=lambda x: x[1])
        sorted_inputs = [input_ for input_, _ in sorted_inputs_with_lengths]
        return sorted_inputs
        
    # TODO: bad practice, fix this into another function
    @torch.no_grad() 
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        return_raw_output: bool = False,
        heads_to_prune: List[int] = [3],
        layers_to_prune: List[int] = [3],
        stat_track: bool = True,
        save_every_n_gens: int = 10,
        prompt_cache_path: str = "tmp_attention",
        use_prompt_cache: bool = True,
    ):

        if config is None:
            config = self.config
        
            
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune, 
                                         stat_track=stat_track)

        
        if not self.use_accelerate: 
            if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
            elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
                model_inputs = inputs
            elif isinstance(inputs, str):
                model_inputs = self.tokenizer(inputs, return_tensors='pt')
            else:
                raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
            seq_len = model_inputs['input_ids'].shape[1]
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=config.get("max_new_tokens", 50),
                temperature=config.get("temperature", 1.0),
                top_k=config.get("top_k", 50),
                top_p=config.get("top_p", 1.0),
                repetition_penalty=config.get("repetition_penalty", 1.0),
                num_return_sequences=config.get("num_return_sequences", 1),
                do_sample=config.get("do_sample", True),
                use_cache=config.get("use_cache", True),
                pad_token_id=self.tokenizer.eos_token_id,
                edit_fn=attention_hooker
            )
            # attention_hooker.__call__.print_calls()
            # attention_hooker.log_stats()
            # attention_hooker.__call__.reset_count()
            # self.stats = attention_hooker.get_stats()
            if return_raw_output:
                return outputs
            else:
                response = self.tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)
                return response
        else:  # TODO: BAD PRACTICE, FIX THIS # https://github.com/huggingface/accelerate/issues/2733
            
            num_return_sequences = config.get("num_return_sequences", 1)
            if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
                batch_size_per_device = config.get('batch_size_per_device')
                formatted_prompts = [inputs[i : i + batch_size_per_device] for i in range(0, len(inputs), batch_size_per_device)]
                model_inputs =[self.tokenizer(formatted_prompt, padding=True, return_tensors='pt')
                               for formatted_prompt in formatted_prompts]
                save_every_n_gens_ = save_every_n_gens * self.distributed_state.num_processes
                split_model_inputs = [model_inputs[i : i + save_every_n_gens_] for i in range(0, len(model_inputs), save_every_n_gens_)]
                print([len(x) for x in split_model_inputs])
            else:
                raise NotImplementedError(f"Invalid input type {type(inputs)}. Must be list of str.")
            
            
                
            if self.distributed_state.is_main_process:
                print()
                print(f"save_every_n_gens: {save_every_n_gens} | len model_inputs: {len(model_inputs)} (should be larger than sene {save_every_n_gens})")
                print(f"len inputs: {len(inputs)}")
                print(f"len split_model_inputs: {len(split_model_inputs)}")
                print(f"prompt cache path: {prompt_cache_path}")
            outputs = []
            for split_count, splitted_model_inputs in enumerate(split_model_inputs):
                if self.distributed_state.is_main_process:
                    print(f"\nsplit_count: {split_count} / len split_model_inputs: {len(splitted_model_inputs)}")
                splitted_model_inputs_cache_path = os.path.join(prompt_cache_path, f"{split_count}_cached_outputs.pkl")
                if os.path.exists(splitted_model_inputs_cache_path) and use_prompt_cache:
                    prompt_cache = self.load_cached_prompts(splitted_model_inputs_cache_path)
                    outputs = prompt_cache.copy()
                    print(f"{splitted_model_inputs_cache_path} exists, loading data from cache. len outputs: {len(outputs)}. Proceed with next generation")
                    self.distributed_state.state.wait_for_everyone()
                    continue
                
                self.distributed_state.state.wait_for_everyone()
                with self.distributed_state.split_between_processes(splitted_model_inputs) as batched_prompts:
                    generated_texts_across_device = []
                    for batch in batched_prompts:
                        torch.cuda.synchronize()
                        batch = {k: v.to(self.distributed_state.device) for k, v in batch.items()}
                        seq_len = batch['input_ids'].shape[1]
                        print(f'process no. {self.distributed_state.process_index} | seq_len: {seq_len} | batch: {batch["input_ids"].shape}')
                        output = self.model.generate(
                            **batch,
                            max_new_tokens=config.get("max_new_tokens", 50),
                            temperature=config.get("temperature", 1.0),
                            top_k=config.get("top_k", 50),
                            top_p=config.get("top_p", 1.0),
                            repetition_penalty=config.get("repetition_penalty", 1.0),
                            num_return_sequences=num_return_sequences,
                            do_sample=config.get("do_sample", True),
                            use_cache=config.get("use_cache", True),
                            pad_token_id=self.tokenizer.eos_token_id,
                            edit_fn=attention_hooker
                        )
                        generated_texts_across_device.extend(self.tokenizer.batch_decode(output[:, seq_len:], skip_special_tokens=True))
                        if self.distributed_state.is_main_process:
                            print(f"len generated_texts_across_device: {len(generated_texts_across_device)} | len batched_prompts: {len(batched_prompts)}  current batch: {batch['input_ids'].shape}")
                    
                    generated_texts_across_device = gather_object(generated_texts_across_device)
                    if self.distributed_state.is_main_process:
                        outputs.extend(generated_texts_across_device)
                        
                self.distributed_state.state.wait_for_everyone()
                if self.distributed_state.is_main_process:
                    print(f"Caching {len(outputs)} outputs to {splitted_model_inputs_cache_path}")
                    self.cache_outputs(outputs, splitted_model_inputs_cache_path)
                self.distributed_state.state.wait_for_everyone()

            if self.distributed_state.is_main_process:
                print(f'len outputs ({len(outputs)}) vs len inputs ({len(inputs)})')
                
                if num_return_sequences == 1:
                    outputs = outputs[:len(inputs)]
                    assert len(outputs) == len(inputs), f"Length mismatch between inputs and outputs: {len(inputs)} != {len(outputs)}"    
                    return outputs
                else:
                    outputs = outputs[:len(inputs) * num_return_sequences]
                    assert len(outputs) == len(inputs) * num_return_sequences, f"Length mismatch between inputs and outputs: {len(inputs)} != {len(outputs)}"    
                    return outputs
                
    
    @torch.no_grad()
    def forward(
        self,
        inputs: Union[str, List[str], Dict],
        heads_to_prune: List[int] = [3],
        layers_to_prune: List[int] = [3],
        stat_track: bool = True,
    ) -> str:
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune,
                                         stat_track=stat_track)
        
        output = self.model(**model_inputs, 
                            edit_fn=attention_hooker,
                            use_cache=False)
        output.logits = output.logits.to('cpu')
        attention_hooker.__call__.print_calls()
        attention_hooker.__call__.reset_count()
        self.stats = attention_hooker.get_stats()

        return output
    
    

if __name__ == '__main__':
    
    os.makedirs("tmp", exist_ok=True)
    model_repo = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    inference_transformers = InferenceTransformers(model_repo, use_auto_model=False)
    text_list = [
        "What is the integral of x^2?",
        "What is the integral of x^3?",
        "What is the integral of x^2?",
        "What is the integral of x^3?",
        "What is the derivative of sin(x) * e^(x^2)?",
        "Solve the equation: 3x^2 - 5x + 2 = 0 using the quadratic formula.",
        "Find the limit of (x^2 + 3x + 2)/(x^2 - 4) as x approaches 2.",
        "What is the Taylor series expansion of cos(x) up to the fourth-degree term?",
        "Evaluate the definite integral of e^(-x^2) from x = -1 to x = 1.",
        "Simplify the expression: (2x^3 - 5x^2 + 4x - 1)/(x - 1) using polynomial division.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "What is the solution of the differential equation dy/dx = x^2 + y^2 with the initial condition y(0) = 1?",
        "Find the Fourier transform of the function f(x) = e^(-|x|).",
        "Evaluate the triple integral of x^2 + y^2 + z^2 over the unit sphere.",
        "What is the determinant of the matrix [[1, 2, 3], [0, -1, 4], [5, 6, 0]]?",
        "Find the Maclaurin series expansion of ln(1 + x) up to the fifth-degree term.",
        "What is the Laplace transform of t*sin(2t)?",
        "Find the general solution of the partial differential equation ∂u/∂t = k∂^2u/∂x^2.",
        "Solve the system of linear equations: 2x + 3y - z = 5, x - 2y + 4z = -3, and 3x + y + 2z = 7.",
        "Evaluate the improper integral of 1/(1 + x^2) from x = -∞ to x = ∞.",
        "What is the volume of the solid generated by revolving the curve y = x^2 around the x-axis from x = 0 to x = 2?",
        "Find the solution to the equation x^4 - 6x^2 + 8 = 0.",
        "What is the inverse of the matrix [[2, 3], [1, 4]]?",
        "Prove that the sum of the first n odd integers is n^2.",
        "Calculate the gradient of the scalar field f(x, y, z) = x^2 + y^2 + z^2 at the point (1, 2, 3).",
        "Find the arc length of the curve y = ln(x) from x = 1 to x = 3."
    ]
    outputs = inference_transformers.generate_non_distributed(text_list, cache_path="tmp/cache_test_3/")
    print(outputs)
    breakpoint()
    
    # data_path = f"notebook/tmp/{model_repo.replace('/', '_')}_generated_outputs_1batch.pkl"
    # # data_path = data_path.replace("1.5B", "7B")
    # with open(data_path, "rb") as in_f:
    #     attention_data_base = pickle.load(in_f)
    
    # attention_data_base['input_ids'] = attention_data_base['input_ids']
    # attention_data_base['labels'] = attention_data_base['labels']
    # default_config = InferenceTransformers.load_config('config/default_generation_config.json')
    
    # model_inputs = inference_transformers.generate(attention_data_base, 
    #                                                return_raw_output=False,
    #                                                heads_to_prune=[],
    #                                                layers_to_prune=[],)
    # print(model_inputs)