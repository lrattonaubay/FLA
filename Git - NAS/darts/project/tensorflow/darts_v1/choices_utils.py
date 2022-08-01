from typing import Any, Dict, List, Optional
from collections import defaultdict


def get_fixed_value(label: str) -> Any:
    ret = get_current_context('fixed')
    try:
        return ret[generate_new_label(label)]
    except KeyError:
        raise KeyError(f'Fixed context with {label} not found. Existing values are: {ret}')


def get_current_context(key: str) -> Any:
    return ContextStack.top(key)

class ContextStack:
    _stack: Dict[str, List[Any]] = defaultdict(list)

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def __enter__(self):
        self.push(self.key, self.value)
        return self

    def __exit__(self, *args, **kwargs):
        self.pop(self.key)

    @classmethod
    def push(cls, key: str, value: Any):
        cls._stack[key].append(value)

    @classmethod
    def pop(cls, key: str) -> None:
        cls._stack[key].pop()

    @classmethod
    def top(cls, key: str) -> Any:
        return cls._stack[key][-1]

def generate_new_label(label: Optional[str]):
    if label is None:
        return "model"
    return label