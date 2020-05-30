import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

class ReforBertForQA(nn.module):
    def __init__(self, config):
        super(ReforBertForQA, self)

        vocab_size = 8007     # vocab 크기
        max_seq_len = 512     # 최대 입력 길이
        embedding_size = 768  # 임베딩 사이
        batch_size = 128      # 학습 시 배치 크기
        depth = 6             # reformer depth
        heads = 8             # reformer heads

        self.num_labels = config.num_labels
        self.reforBert = ReforBertLM(
                            num_tokens=vocab_size,
                            dim=embedding_size,
                            depth=depth,
                            heads=heads,
                            max_seq_len=max_seq_len,
                            causal=True
                        )
        self.qa_output = nn.Linear(embedding_size, self.num_labels)

        torch.nn.init.xavier_uniform_(self.qa_output.weight)
