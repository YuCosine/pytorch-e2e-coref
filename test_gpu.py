import time
import torch

a = torch.zeros(3)
start = time.time()
a = a.cuda()
print(f'cuda() costs {time.time() - start}s')
