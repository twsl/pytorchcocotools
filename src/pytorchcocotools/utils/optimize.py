from functools import wraps
from typing import Any

import torch


# Define the no_grad decorator
def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper
