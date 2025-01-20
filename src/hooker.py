import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict
import torch 
import numpy as np
from collections import defaultdict


def count_decorator(func):
    """
    A decorator to count how many times a function has been called.
    """
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0

    def reset_count():
        wrapper.call_count = 0

    def print_calls():
        print(f"{func.__name__} has been called {wrapper.call_count} time(s).")

    wrapper.reset_count = reset_count
    wrapper.print_calls = print_calls
    return wrapper


class BaseHooker:
    def __init__(self, layer_list: List[int], stat_track: bool = True) -> None:
        self.attention = None
        self.layer_list = layer_list
        self.stats: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {'mean': [], 'var': [], 'norm': []})
        self.current_layer = None
        self.stat_track = stat_track
        
    def print_stats(self):
        if not self.stat_track:
            print("Stat tracking is disabled.")
            return
        
        for layer_idx, stats in self.stats.items():
            # aggregate the mean and variance of the attention output
            means = np.array(stats['mean'])
            variances = np.array(stats['var'])
            print(f"Layer {layer_idx}:")
            print(f"Mean: {np.mean(means, axis=0)}")
            print(f"Variance: {np.mean(variances, axis=0)}")
            print()
            
    def get_stats(self):
        return self.stats
    
    def track_stats(self, attn_output):
        mean = attn_output.mean(dim=(0, 1, 3)).detach().cpu().float().numpy()
        var = attn_output.var(dim=(0, 1, 3)).detach().cpu().float().numpy()
        norm = attn_output.norm(dim=3).norm(dim=1).norm(dim=0).detach().cpu().float().numpy()
        self.stats[self.current_layer]['mean'].append(mean)
        self.stats[self.current_layer]['var'].append(var)
        self.stats[self.current_layer]['norm'].append(norm)
        
    @count_decorator
    def __call__(self, attn_output):
        if self.current_layer not in self.layer_list:
            return attn_output
        
        if self.stat_track:
            self.track_stats(attn_output)
        
        self.attention = attn_output.detach().cpu()
        return attn_output
    
class ZeroOutHooker(BaseHooker):
    def __init__(self, head_indices: List[int], layer_list: List[int], stat_track: bool = True) -> None:
        super().__init__(layer_list, stat_track)
        self.head_indices = head_indices
    
    
    @count_decorator
    def __call__(self, attn_output):
        if self.current_layer not in self.layer_list:
            return attn_output
        
        if not isinstance(attn_output, torch.Tensor):
            raise TypeError("attn_output must be a torch.Tensor")
        if len(attn_output.shape) != 4:
            raise ValueError("attn_output must have shape (batch_size, seq_len, num_attention_head, head_dim)")
        if not all(isinstance(idx, (int, np.integer)) for idx in self.head_indices):
            raise ValueError("head_indices must be a list of integers")
       
        if self.stat_track:
            self.track_stats(attn_output)
            
        batch_size, seq_len, num_attention_head, head_dim = attn_output.shape
        if any(idx < 0 or idx >= num_attention_head for idx in self.head_indices):
            raise ValueError("head_indices contains invalid head indices")

        mask = torch.ones(num_attention_head, 
                        dtype=attn_output.dtype, 
                        device=attn_output.device)
        mask[self.head_indices] = 0  
        mask = mask.view(1, 1, num_attention_head, 1)  
        attn_output = attn_output * mask  

        return attn_output

