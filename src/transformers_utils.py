import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from typing import Union, List, Dict
import json
from pathlib import Path
import logging
import torch 
import transformers 
from functools import partial
import numpy as np
from hooker import BaseHooker, ZeroOutHooker


class InferenceTransformers:
    def __init__(self, model_repo: str, config: Dict = None, use_auto_model: bool = True):
        self.model_repo = model_repo
        
        self.model_name = model_repo.split('/')[-1]
        self.default_generation_config_path = "config/default_generation_config.json"
        if config is not None:
            self.config = config
        else:
            self.config = self.load_config(self.default_generation_config_path)
        self.model, self.tokenizer = self.init_model_tokenizer(model_repo, 
                                                               use_auto_model=use_auto_model,
                                                               attention_implementation=self.config.get("attention_implementation", 'eager'))
    
    @staticmethod
    def init_model_tokenizer(model_repo: str, 
                             use_auto_model: bool = True,
                             attention_implementation: str = 'eager'):
        
        if use_auto_model:
            model = AutoModelForCausalLM.from_pretrained(model_repo, 
                                                        torch_dtype=torch.bfloat16, 
                                                        device_map='auto',
                                                        attn_implementation=attention_implementation)
                                                        #  attn_implementation='flash_attention_2')
        else:
            if 'qwen' in model_repo.lower():
                from mod_llm.qwen2.modeling_qwen2 import Qwen2ForCausalLM
                model = Qwen2ForCausalLM.from_pretrained(model_repo, 
                                                        torch_dtype=torch.bfloat16, 
                                                        device_map='auto',
                                                        attn_implementation=attention_implementation)
            else:
                raise ValueError(f"Model {model_repo} not supported. or set use_auto_model=True instead")
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        return model, tokenizer
    
    @staticmethod
    def save_config(config: Dict, filepath: str = "config/default_generation_config.json"):
        """Save generation configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    @staticmethod    
    def load_config(filepath: str = "config/default_generation_config.json") -> Dict:
        """Load generation configuration from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Config file not found at {filepath}.")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        return_raw_output: bool = False,
    ):
        
        if config is None:
            config = self.config
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        output = self.model.generate(
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
        )
        
        if return_raw_output:
            return output
        else:
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            return response
        
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
        
        output = self.model(**model_inputs)
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
        attention_hooker.print_stats()  
        print(f"Attention hooker has been called {attention_hooker.__call__.call_count} time(s).")
        return attention_hooker.attention
    
    
class IntervenableTransformers(InferenceTransformers):
    def __init__(self, model_repo: str, config: Dict = None, use_auto_model: bool = True):
        super().__init__(model_repo, config, use_auto_model)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        config: Dict = None,
        return_raw_output: bool = False,
        heads_to_prune: List[int] = [3],
        layers_to_prune: List[int] = [3],
    ):
        
        if config is None:
            config = self.config
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = inputs
        elif isinstance(inputs, str):
            model_inputs = self.tokenizer(inputs, return_tensors='pt')
        else:
            raise ValueError(f"Invalid input type {type(inputs)}. Must be str, list of str, or dict.")
        
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune, 
                                         stat_track=False)
        
        output = self.model.generate(
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
        attention_hooker.__call__.print_calls()
        attention_hooker.print_stats()
        
        if return_raw_output:
            return output
        else:
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            return response
    
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
        
        attention_hooker = ZeroOutHooker(head_indices=heads_to_prune, 
                                         layer_list=layers_to_prune,
                                         stat_track=stat_track)
        
        output = self.model(**model_inputs, 
                            edit_fn=attention_hooker,
                            use_cache=False)
        output.logits = output.logits.to('cpu')
        attention_hooker.__call__.print_calls()
        self.stats = attention_hooker.get_stats()
        # attention_hooker.print_stats()
        return output
    
    
    
if __name__ == '__main__':
    # log_folder = '../log'
    # os.makedirs(log_folder, exist_ok=True)
    # log_file = os.path.join(log_folder, 'transformers_utils.log')
    # logging.basicConfig(
    #     filename=log_file,
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    import pickle
    import os 
    os.makedirs("tmp", exist_ok=True)
    model_repo = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    data_path = f"../notebook/tmp/{model_repo.replace('/', '_')}_generated_outputs_1batch.pkl"
    data_path = data_path.replace("1.5B", "7B")
    with open(data_path, "rb") as in_f:
        attention_data_base = pickle.load(in_f)
    
    attention_data_base['input_ids'] = attention_data_base['input_ids']
    attention_data_base['labels'] = attention_data_base['labels']
    inference_transformers = IntervenableTransformers(model_repo, 
                                                   use_auto_model=False)
    
    default_config = InferenceTransformers.load_config('config/default_generation_config.json')
    output = inference_transformers.forward(inputs=attention_data_base,
                                            heads_to_prune=[3],
                                            layers_to_prune=[3], 
                                            stat_track=True)
    # output = inference_transformers.get_attention_at_layers(inputs=attention_data_base, layers_to_prune=[3])
    
    # print(output)
    # text = "Square root of 64 is "
    # output = inference_transformers.generate(text)
    # print(output)
    # breakpoint()
    # logging.info('Testing single text ')
    # logging.info(inference_transformers.generate(text, return_raw_output=False))
    # logging.info(inference_transformers.generate(text, return_raw_output=True).shape)
    # logging.info('Testing multiple texts ')
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=False))
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=True).shape)
    # logging.info('Testing tokenized text ')
    # model_inputs = inference_transformers.tokenizer(text, return_tensors="pt")
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)
    # logging.info('Testing batch tokenized text ')
    # model_inputs = inference_transformers.tokenizer(["Hello, I am a", "you are dead to me!"], padding=True, return_tensors="pt")
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    # logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)