from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from typing import Union, List, Dict
import json
from pathlib import Path

class InferenceTransformers:
    def __init__(self, model_repo: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model_name = model_repo.split('/')[-1]
    
    def save_config(self, config: Dict, filepath: str = "config/default_generation_config.json"):
        """Save generation configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
            
    def load_config(self, filepath: str = "config/default_generation_config.json") -> Dict:
        """Load generation configuration from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Config file not found at {filepath}.")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_activation_at_layer(self, layer: int, text: str):
        pass 
    
    def forward(self, ):
        pass
    
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        max_length: int = 50
    ) -> str:
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            input_ids = self.tokenizer(inputs, padding=True, return_tensors='pt')
        elif isinstance(inputs, dict):
            input_ids = self.tokenizer(inputs, return_tensors='pt')
        else:
            input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
            
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def generate_tracedict(self, ):
        pass
        
    def forward_tracedict(self, ):
        pass
    
    
if __name__ == '__main__':
    model_repo = "gpt2"
    inference_transformers = InferenceTransformers(model_repo)
    text = "Hello, I am a"
    print(inference_transformers.generate(text, max_length=50))