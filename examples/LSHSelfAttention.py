import torch
from Reformer.reformer_pytorch import LSHSelfAttention

attn = LSHSelfAttention(
    dim = 128,
    heads = 8, #수 헤더 크기
    bucket_size = 64, # 버켓 사이즈
    n_hashes = 8, # 해시 개
    causal = False
)

x = torch.randn(10, 1024, 128)
y = attn(x) # (10, 1024, 128)

print('LSH Sefl Attention')
print('x size: ',x.size())
print('x shape: ',x.shape)
print('x : ',x)
print('y: ',y)
