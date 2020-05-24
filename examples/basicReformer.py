# should fit in ~ 5gb - 8k embeddings

import torch
from Reformer.reformer_pytorch import Reformer

model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
)

x = torch.randn(1, 8192, 512)
y = model(x) # (1, 8192, 512)

print('y: ',y)