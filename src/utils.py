import numpy as np
from functools import wraps
import time 

def time_decorator(func):
    @wraps(func)  # This preserves the original function's metadata (e.g., name, docstring)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result  # Return the result of the wrapped function
    return wrapper


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