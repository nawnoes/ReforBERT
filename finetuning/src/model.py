import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

class ReforbertForSequenceClassification(ReforBertLM)