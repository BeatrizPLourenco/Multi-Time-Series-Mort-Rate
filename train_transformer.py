#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:48:24 2022

@author: beatrizlourenco
"""

import torch 
import numpy as np
from tqdm import tqdm
from random import shuffle, choice
from torch import Tensor
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bptt = 5
def get_batch(source: Tensor, i: int):

    return source[0][i, :,:,:], source[1][i, :,:,:], source[2][i, :,:,:]

def train(model, train_data, tgt_mask, src_mask, epoch, optimizer,lr, criterion ) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt

    for batch_nb in (range(train_data[0].size(0))):
        x, y_input, y_expected  = get_batch(train_data, batch_nb)
        seq_len = x.size(2)
        num_batches = x.size(0)
        print(seq_len, num_batches)

        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(x, y_input, src_mask, tgt_mask)
        loss = criterion(output, y_expected)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_nb % log_interval == 0 and batch_nb > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch_nb:5d}/{num_batches:5d} batches | '
                f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model, eval_data, tgt_mask,  src_mask, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for batch_nb in range(0, eval_data[0].size(0) - 1):
            x_val, y_val_input, y_val_expected = get_batch(eval_data, batch_nb)
            seq_len = x_val.size(2)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(x_val, y_val_input, src_mask, tgt_mask)
            total_loss += seq_len * criterion(output, y_val_expected).item()
    return total_loss / ( (x_val.size(0)*x_val.size(1)) - 1)
