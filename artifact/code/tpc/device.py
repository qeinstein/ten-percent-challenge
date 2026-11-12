import torch
def best(): return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
