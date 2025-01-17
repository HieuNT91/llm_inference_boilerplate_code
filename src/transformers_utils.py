import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from typing import Union, List, Dict
import json
from pathlib import Path
import logging
import torch 
import transformers 

class InferenceTransformers:
    def __init__(self, model_repo: str):
        self.model_repo = model_repo
        self.model, self.tokenizer = self.init_model_tokenizer(model_repo)
        
        self.model_name = model_repo.split('/')[-1]
        self.default_generation_config_path = "config/default_generation_config.json"
    
    @staticmethod
    def init_model_tokenizer(model_repo: str):
        model = AutoModelForCausalLM.from_pretrained(model_repo, 
                                                     torch_dtype=torch.bfloat16, 
                                                     device_map='auto', 
                                                     attn_implementation='flash_attention_2')
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
        return_raw_output: bool = False
    ) -> str:
        
        if config is None:
            config = self.load_config(self.default_generation_config_path)
        
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
            max_length=config.get("max_length", 50),
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 1.0),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            num_return_sequences=config.get("num_return_sequences", 1),
            do_sample=config.get("do_sample", True),
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        if return_raw_output:
            return output
        else:
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            return response
        
        # return self.tokenizer.batch_decode(output, skip_special_tokens=True)
    
    def get_activation_at_layer(self, layer: int, text: str):
        pass 
    
    def forward(self, ):
        pass
    
    def generate_tracedict(self, ):
        pass
        
    def forward_tracedict(self, ):
        pass
    
    
if __name__ == '__main__':
    log_folder = '../log'
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, 'transformers_utils.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    model_repo = "meta-llama/Meta-Llama-3-8B-Instruct"
    inference_transformers = InferenceTransformers(model_repo)
    default_config = InferenceTransformers.load_config('config/default_generation_config.json')
    text = "Hello, I am a"
    # logging.info('Testing single text ')
    # logging.info(inference_transformers.generate(text, return_raw_output=False))
    # logging.info(inference_transformers.generate(text, return_raw_output=True).shape)
    # logging.info('Testing multiple texts ')
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=False))
    # logging.info(inference_transformers.generate(["Hello, I am a", "you are dead to me!"], return_raw_output=True).shape)
    logging.info('Testing tokenized text ')
    model_inputs = inference_transformers.tokenizer(text, return_tensors="pt")
    logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)
    logging.info('Testing batch tokenized text ')
    model_inputs = inference_transformers.tokenizer(["Hello, I am a", "you are dead to me!"], padding=True, return_tensors="pt")
    logging.info(inference_transformers.generate(model_inputs, return_raw_output=False))
    logging.info(inference_transformers.generate(model_inputs, return_raw_output=True).shape)