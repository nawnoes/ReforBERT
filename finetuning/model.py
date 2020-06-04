import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from reformer.model import ReforBertLM, Reformer

class ReforBertForQA(nn.Module):
    def __init__(self, config):
        super(ReforBertForQA, self)

        vocab_size = 8007     # vocab 크기
        max_seq_len = 512     # 최대 입력 길이
        embedding_size = 768  # 임베딩 사이
        batch_size = 128      # 학습 시 배치 크기
        depth = 6             # reformer depth
        heads = 8             # reformer heads
        device ="cpu"         # cpu or cuda


        self.num_labels = config.num_labels
        self.reforBert = ReforBertLM(
                            num_tokens=vocab_size,
                            dim=embedding_size,
                            depth=depth,
                            heads=heads,
                            max_seq_len=max_seq_len,
                            causal=True ) # model(inputs, segments)

        self.qa_output = nn.Linear(embedding_size, self.num_labels)

    def forward(
                self,
                input_ids=None,
                segments_ids = None
                ):
        # 1. reforBert에 대한 입력
        outputs = self.reforBert(input_ids,segments_ids)

        # 2. reforBert 출력에 대해 classification 위해 Linear 레이어 통과
        logits = self.qa_outputs(sequence_output)

        # 3. start logits, end_logits 구하기
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[1:]
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
