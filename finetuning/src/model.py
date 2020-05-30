import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

class ReforBertForQA(nn.Module):
    def __init__(self, config):
        super(ReforBertForQA, self)

        self.num_labels = config.num_labels
        self.reforBert = ReforBertLM()
