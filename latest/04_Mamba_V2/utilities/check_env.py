import torch
print('CUDA:', torch.cuda.is_available())
try:
    from mamba_ssm import Mamba
    print("mamba: YES")
except:
    print("mamba: FALLBACK (GRU)")
