import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

BertLayerNorm = nn.LayerNorm

class ReforBertForSequenceClassification(nn.Module):
    def __init__( self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.device = torch.device(config.device)

        self.reforBert = ReforBertLM(
                            num_tokens= config.vocab_size,
                            dim= config.embedding_size,
                            depth= config.depth,
                            heads= config.heads,
                            max_seq_len= config.max_seq_len,
                            causal=True ) # model(inputs, segments)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_pretrained(self, pretrained_model_path):
        if os.path.isfile(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            self.reforBert.load_state_dict(checkpoint['model_state_dict'])

    def forward(
                self,
                input_ids=None,
                token_type_ids = None, # 세그멘트 id
                labels=None,
    ):
        # 1. reforBert에 대한 입력
        outputs, _, _ = self.reforBert(input_ids,token_type_ids)

        pooled_output = outputs[0][:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)# + discriminator_hidden_states[1:]  # add hidden states and attention if they are here

        if labels is not None:
          if self.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
          else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          outputs = (loss,) + outputs

        return outputs  # (loss), logits, 뒷부분 추가 X (hidden_states), (attentions)
