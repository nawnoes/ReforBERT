import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections
from tqdm import tqdm, trange
import json
from random import random, randrange, randint, shuffle, choice
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

""" pretrain 데이터셋"""
class PretrainDataSet(torch.utils.data.Dataset):
  """
  데이터로더에 사용하기 위한 데이터 셋

  is_next: tokens_a와 tokens_b가 연속된 문장인지 여부
  tokens: 문장들의 tokens
  segment: tokens_a(0)와 tokens_b(1)을 구분하기 위한 값
  mask_idx: tokens 내 mask index
  mask_label: tokens 내 mask 된 부분의 정답
  """
  def __init__(self, vocab, infile):
    self.vocab = vocab
    self.labels_cls = []
    self.labels_lm = []
    self.sentences = []
    self.segments = []

    line_cnt = 0
    with open(infile, "r") as f:
      for line in f:
        line_cnt += 1

    with open(infile, "r") as f:
      for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
        instance = json.loads(line)
        self.labels_cls.append(instance["is_next"])
        sentences = [vocab.piece_to_id(p) for p in instance["tokens"]]
        self.sentences.append(sentences)
        self.segments.append(instance["segment"])
        mask_idx = np.array(instance["mask_idx"], dtype=np.int)
        mask_label = np.array([vocab.piece_to_id(p) for p in instance["mask_label"]], dtype=np.int)
        label_lm = np.full(len(sentences), dtype=np.int, fill_value=-1)
        label_lm[mask_idx] = mask_label
        self.labels_lm.append(label_lm)

  def __len__(self):
    assert len(self.labels_cls) == len(self.labels_lm)
    assert len(self.labels_cls) == len(self.sentences)
    assert len(self.labels_cls) == len(self.segments)
    return len(self.labels_cls)

  def __getitem__(self, item):
    return (torch.tensor(self.labels_cls[item]),
            torch.tensor(self.labels_lm[item]),
            torch.tensor(self.sentences[item]),
            torch.tensor(self.segments[item]))


""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    """
    배치 단위로 데이터 처리를 위한 collate_fn

    :param inputs:
    :return: batch
    """
    labels_cls, labels_lm, inputs, segments = list(zip(*inputs))

    # LM의 라벨의 길이가 같아지도록, 짧은 문장에 대해 padding 값-1 추가
    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    # inputs의 길이가 같아지도록 짧은 문장에 대해 padding 값 0 추가 이때 padding은 vocab 만들기 시, pad_id = 0으로 지정한 값
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    # segments에 대한 값도 짧은 문장에 대해 padding값 0 추가
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels_cls, dim=0), # 길이가 고정 1이므로, stack 함수를 통해 torch tensor로 변환
        labels_lm,
        inputs,
        segments
    ]
    return batch


""" pretrain 데이터 로더 """
def pretrain_data_loader(vocab, data_dir, batch_size = 128):
  dataset = PretrainDataSet(vocab, f"{data_dir}/kowiki_bert_0.json")
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrin_collate_fn)

  return data_loader

