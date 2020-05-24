import os
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn


from tqdm import tqdm
from transformers import BertTokenizer
from fairseq.optim.adafactor import Adafactor

from gluonnlp.data import SentencepieceTokenizer
from ReforBERT.util.tokenizer import koBERTTokenizer
from ReforBERT.reformer import Reformer, ReformerLM
from ReforBERT.util.pretrain import kobert_mask_tokens

from ReforBERT.dataloader.common import build_dataloaders
from ReforBERT.dataloader.wiki import WikiDataset

class ReformerLMTrainer(object):
  """
  Reformer의 Language Model을 학습시키기 위한 클래스

  :param dataset: (torch.utils.data.Dataset) containing all of the data you wish to utilize during training.
  :param model: (reformer_pytorch.Reformer)
  :param tokenizer: 한국어 버전에서는 기존에 Kobert에서 사용한 vocab과 tokenizer를 사용
                   원본에서는 (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
  :param device: provide manual device placement. If None, will default to cuda:0 if available.
  :param tb_writer: (bool) Whether to write to tensorboard or not.
  :param tb_dir: (str) Where to write TB logs to.
  :param log_dir: (str) Where to write generic logs to.
  """
  def __init__(self,
               dataset,
               model,
               tokenizer,
               vocab,
               device=None,
               train_batch_size=8,
               eval_batch_size=None,
               tb_writer=True,
               tb_dir='./tb_logs',
               log_dir='./logs'):
    self.dataset = dataset
    self.model = model
    self.tokenizer = tokenizer
    self.vocab = vocab
    self.device = device
    self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    self.train_batch_size = train_batch_size
    self.eval_batch_size = eval_batch_size
    self.tb_writer = tb_writer
    self.log_dir = log_dir

    # 토크나이저 설정
    if tokenizer is None: #
      self.tokenizer = koBERTTokenizer() # KoBERT 토크나이저
    # 디바이스 설정
    if device is None:
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 평가 배치 결정
    if eval_batch_size is None:
      self.eval_batch_size = train_batch_size
    # 텐서보드 사용 설
    if tb_writer:
      from torch.utils.tensorboard import SummaryWriter
      self.writer = SummaryWriter(log_dir=tb_dir)
    # 로깅을 위한 설
    logging.basicConfig(filename=f'{log_dir}/{datetime.now().date()}.log', level=logging.INFO)

  def train(self,
            epochs,
            train_dataloader,
            eval_dataloader,
            log_steps,
            ckpt_steps,
            ckpt_dir=None,
            gradient_accumulation_steps=1):
    """
    Reformer LM 학습

    :param epochs:
    :param train_dataloader:
    :param eval_dataloader:
    :param log_steps: 로그가 찍히는 스텝
    :param ckpt_steps: 체크포인트 저장 스템
    :param ckpt_dir: 체크포인드 저장 경로
    :param gradient_accumulation_steps: gradient accumulation 옵
    :return: 총 학습 step 수, 총 Loss, 모
    """

    optimizer = Adafactor(self.model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    losses = {}
    global_steps = 0
    local_steps = 0
    step_loss = 0.0

    # 체크포인트 경로 설정
    # 경로가 있는경우 학습을 위해 다시 불러온다.
    # 수정 필요.
    if ckpt_dir is not None:
      assert os.path.isdir(ckpt_dir)
      try:
        logging.info(f'{datetime.now()} | Continuing from checkpoint...')
        self.model.load_state_dict(torch.load(f'{ckpt_dir}/model_state_dict.pt', map_location=self.device))
        optimizer.load_state_dict(torch.load(f'{ckpt_dir}/optimizer_state_dict.pt'))

      except Exception as e:
        logging.info(f'{datetime.now()} | No checkpoint was found | {e}')

    # 모델 학습 모드
    self.model.train()

    # 다중 지피유 사용인 경
    if self.n_gpu > 1:
      self.model = nn.DataParallel(self.model)
      logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

    self.model.to(self.device)
    # 학습 정보  출력
    logging.info(f'{datetime.now()} | Moved model to: {self.device}')
    logging.info(f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
    logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
    logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

    # ReformerLM 학습
    for epoch in tqdm(range(epochs), desc='Epochs', position=0):
      logging.info(f'{datetime.now()} | Epoch: {epoch}')
      for step, batch in tqdm(enumerate(train_dataloader),
                              desc='Epoch Iterator',
                              position=1,
                              leave=True,
                              total=len(train_dataloader)):
        # 배치별 데이터 학습
        for data in batch:
          inputs = self._tokenize_input_ids(data, pad_to_max_length=True)
          inputs, labels = kobert_mask_tokens(inputs) # *** 여러 문장이 들어와도 괜찮은지 확인 필요
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          output = self.model(inputs)

          # only calculating loss on masked tokens
          loss_mx = labels != self.vocab.to_indices(self.vocab.mask_token) # 마스킹 토큰 아이디로 변경
          output = output[loss_mx].view(-1, len(self.tokenizer.vocab))
          labels = labels[loss_mx].view(-1)

          loss = loss_fn(output, labels)

          if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps

          loss.backward()
          optimizer.step()
          self.model.zero_grad()

          step_loss += loss.item()
          losses[global_steps] = loss.item()
          local_steps += 1
          global_steps += 1

          if global_steps % log_steps == 0:
            if self.tb_writer:
              self.writer.add_scalar('Train/Loss', step_loss / local_steps, global_steps)
              self.writer.close()
            logging.info(
              f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')

            with open(f'{self.log_dir}/train_results.json', 'w') as results_file:
              json.dump(losses, results_file)
              results_file.close()
            step_loss = 0.0
            local_steps = 0

          if global_steps % ckpt_steps == 0:
            # evaluating before every checkpoint
            self.evaluate(eval_dataloader)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
            torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

            logging.info(f'{datetime.now()} | Saved checkpoint to: {ckpt_dir}')

    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
    torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

    return self.model

if __name__=='__main__':
  # 데이터 셋
  data_path = '/Users/a60058238/Desktop/dev/wokspace/nlp/Data/kowiki'
  dataset = WikiDataset(path='D:/data/enwiki')

  # 기존 BERT 토크나이저
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

  # KoBERT 토크나이저
  tok_path = koBERTTokenizer()
  sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

  vocab_size = 8002 # vocab 크기
  max_seq_len = 512 # 최대 입력 길이
  embedding_size = 768 # 임베딩 사이
  batch_size = 4 # 학습 시 배치 크기

  # Refomer Language Model 생
  model = ReformerLM(
    num_tokens=vocab_size,
    dim=embedding_size,
    depth=6,
    heads=8,
    max_seq_len=max_seq_len,
    causal=True
  )
  trainer = ReformerLMTrainer(dataset, model, tokenizer, train_batch_size=32, eval_batch_size=32)
  train_dataloader, eval_dataloader = build_dataloaders(train_test_split=0.90)
  model = trainer.train(epochs=3,
                        train_dataloader=train_dataloader,
                        eval_dataloader=eval_dataloader,
                        log_steps=10,
                        ckpt_steps=100,
                        ckpt_dir='./checkpoint',
                        gradient_accumulation_steps=1)
