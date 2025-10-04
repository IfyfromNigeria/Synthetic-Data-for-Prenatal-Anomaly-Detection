import os, random, json
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_history_json(history: dict, path: str):
    with open(path, "w") as f:
        json.dump(history, f)

def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"
