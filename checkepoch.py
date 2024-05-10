import torch

a = torch.load('/remote-home/mengxichen/UniBrain-lora/Pretrain/output_missft/output_mod3v1/best_val.pth')
print(a['epoch'])