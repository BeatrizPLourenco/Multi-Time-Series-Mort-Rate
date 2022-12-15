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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def train(model, src, trg,tgt_mask, src_mask, num_epochs, optimizer, criterion ):
    all_losses = []
    src_shape = src.shape
    trg_shape = trg.shape
    
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch: {epoch}')
        loss_agg = 0
        
        for idx in tqdm(range(len(trg))):
            target_line_tensor = trg[idx]
            input_line_tensor = src[idx]  
            target_line_tensor = torch.reshape(target_line_tensor, (1,trg_shape[1],trg_shape[2]))
            input_line_tensor = torch.reshape(input_line_tensor, (1,src_shape[1],src_shape[2]))

            optimizer.zero_grad()
            outputs = model(input_line_tensor, target_line_tensor, src_mask, tgt_mask)

            loss = torch.stack(
                [ criterion(o_i, t_i) for o_i ,t_i in zip(outputs, target_line_tensor)]
            ).sum()

            loss.backward()
            optimizer.step()
            
            loss_agg += loss.item()/ input_line_tensor.size(0)
            
            all_losses.append(loss_agg)
            
                    
    return all_losses



