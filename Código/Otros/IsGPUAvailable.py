import torch
device = 'cuda' if torch.cuda.is_available() else 'CPU'
print(device)