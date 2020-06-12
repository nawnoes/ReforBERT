import os
import json
import logging
import numpy as np
from datetime import datetime
import math
from random import random, randrange, randint, shuffle, choice
import matplotlib.pyplot as plt
import json
import pandas as pd
from IPython.display import display
from tqdm import tqdm, tqdm_notebook, trange
import sentencepiece as spm
import wget

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import dataloader

from reformer.model import ReforBertLM, ReformerLM
from dataloader.kowiki import PretrainDataSet, pretrin_collate_fn
from util.vocab import load_vocab





""" random seed """
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" init_process_group """
def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


""" destroy_process_group """
def destroy_process_group():
    dist.destroy_process_group()

""" 모델 epoch 학습 """
def train_epoch(device, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader, train_save_step, train_step = 0):
    losses = []
    train_start_index = train_step+1 if train_step != 0 else 0
    total_train_step = len(train_loader) - train_start_index
    model.train()

    with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader, train_start_index):
            if i >= total_train_step:
                torch.save({
                    'epoch': epoch+1,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss,  # Loss 저장
                    'train_step': 0,  # 현재 진행한 학습
                    'total_train_step': 0 # 현재 epoch에 학습 할 총 train step
                }, save_pretrain)
                break
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs, logits_cls, logits_lm = model(inputs, segments)


            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))

            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            if i % train_save_step == 0:
                torch.save({
                    'epoch': epoch,                                   # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),           # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),   # 옵티마이저 저장
                    'loss': loss,                                     # Loss 저장
                    'train_step': i,                                  # 현재 진행한 학습
                    'total_train_step': len(train_loader)             # 현재 epoch에 학습 할 총 train step
                }, save_pretrain)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)



if __name__ == '__main__':
    # Data 및 Vocab 경로
    data_path = "/Users/a60058238/Desktop/dev/workspace/ReforBERT/data/kowiki"
    checkpoint_path ="../checkpoint"
    save_pretrain = f"{checkpoint_path}/save_reforBERT_pretrain.pth"
    vocab_path = "/Users/a60058238/Desktop/dev/workspace/ReforBERT/data/kowiki/kowiki.model"

    vocab = spm.SentencePieceProcessor()
    vocab = load_vocab(vocab_path)

    count = 10            # 학습 데이터 분할 크기 kowiki_bert_{}.json
    learning_rate = 5e-5  # Learning Rate
    n_epoch = 20          # Num of Epoch
    batch_size = 128      # 배치 사이즈
    device ="cpu"         # cpu or cuda

    vocab_size = 8007     # vocab 크기
    max_seq_len = 512     # 최대 입력 길이
    embedding_size = 768  # 임베딩 사이
    batch_size = 1      # 학습 시 배치 크기
    depth = 6             # reformer depth
    heads = 8             # reformer heads

    train_save_step = 100 # 학습 저장 주기

    # pretrain 데이터 로더
    dataset = PretrainDataSet(vocab, f"{data_path}/kowiki_bert_test.json")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=pretrin_collate_fn)

    # Refomer Language Model 생성
    model = ReforBertLM(
        num_tokens=vocab_size,
        dim=embedding_size,
        depth=depth,
        heads=heads,
        max_seq_len=max_seq_len,
        causal=True
    )
    model.to(device)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch, best_loss, train_step = 0, 0, 0
    if os.path.isfile(save_pretrain):
        checkpoint = torch.load(save_pretrain)
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        train_step =  checkpoint['train_step']
        total_train_step =  checkpoint['total_train_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
        # best_epoch += 1

    losses = []
    offset = best_epoch
    for step in range(n_epoch):
        epoch = step + offset
        if 0 < step:
            del train_loader
            dataset = PretrainDataSet(vocab, f"{data_path}/kowiki_bert_{epoch % count}.json")
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                       collate_fn=pretrin_collate_fn)
            train_step = 0
            total_train_step = 0
        loss = train_epoch(device, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader, train_save_step, train_step)
        losses.append(loss)

    # data
    data = {
        "loss": losses
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


