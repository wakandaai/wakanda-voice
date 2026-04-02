import torch

def detect_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype_str(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "float32"

def get_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32