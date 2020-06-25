import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

BertLayerNorm = nn.LayerNorm


class ReforBertForQA(nn.Module):
    def __init__( self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.reforBert = ReforBertLM(
                            num_tokens= config.vocab_size,
                            dim= config.embedding_size,
                            depth= config.depth,
                            heads= config.heads,
                            max_seq_len= config.max_seq_len,
                            causal=True ) # model(inputs, segments)
        self.device = torch.device(config.device)
        self.qa_outputs = nn.Linear(config.embedding_size, 2)
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
                start_positions=None,
                end_positions=None,
                ):
        # 1. reforBert에 대한 입력
        outputs, _, _ = self.reforBert(input_ids,token_type_ids)

        # 2. reforBert 출력에 대해 classification 위해 Linear 레이어 통과
        logits = self.qa_outputs(outputs)

        # 3. start logits, end_logits 구하기
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # (start_logits, end_logits,) #+ outputs[1:]
        # 에러 발생 reforbert에서 출력결과가 튜플이 아니라서 발생
        outputs = (start_logits, end_logits,) #+ outputs[1:]
        if start_positions is not None and end_positions is not None:

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 손실 함수
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # 시작과 끝 포지션 예측값에 대한 손실값
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 토탈 손실값
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
