import torch
from Reformer.reformer_pytorch import ReformerLM

DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

encoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = DE_SEQ_LEN,
    fixed_position_emb = True,
    return_embeddings = True # return output of last attention layer
)

decoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = EN_SEQ_LEN,
    fixed_position_emb = True,
    causal = True
)

x  = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long()
print('reformer input: ',x.size())
yi = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long()
print('reformer label: ',yi.size())


enc_keys = encoder(x)               # (1, 4096, 1024)
yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)
print('reformer decoder output: ',yo)