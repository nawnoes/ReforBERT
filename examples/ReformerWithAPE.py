"""
리포머 팀에 의하면 포지션 인코딩에서 Axial Position Encoding 사용.
"""

import torch
from Reformer.reformer_pytorch import ReformerLM

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 8192,
    ff_chunks = 8,
    attn_chunks = 2,
    causal = True,
    axial_position_emb = True,
    axial_position_shape = (128, 64),  # the shape must multiply up to the max_seq_len (128 x 64 = 8192)
    axial_position_dims = (512, 512)   # the dims must sum up to the model dimensions (512 + 512 = 1024)
)

x = torch.randint(0, 20000, (1, 8192)).long()
y = model(x) # (1, 8192, 20000)

print('Reformer APE Test')
print('x size: ', x.size())
print('y size: ', y.size())