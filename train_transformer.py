
"""
Created on Tue Oct 18 14:48:24 2022

@author: beatrizlourenco
"""

import torch 
import numpy as np
from tqdm import tqdm
from random import shuffle, choice
from torch import Tensor
import math, time, copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(source: Tensor, i: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gets specific batch in a set.

    Returns:
        Tensor with desired batch.
    """
    
    return source[0][i, :,:,:], source[1][i, :], source[2][i, :], source[3][i, :]



def train(
    model: torch.nn.Module,
    batch_size: int,
    epoch: int,
    train_data: Tensor,
    tgt_mask: Tensor,
    xe_mask: Tensor,  
    optimizer: torch.optim, 
    lr: float = 0.05,
    criterion = torch.nn.MSELoss) -> None:

    """Trains the model through 1 epoch.

    Args:
        model: model to train.
        batch_size: size of each batch.
        epoch: epoch number.
        train_data: data for training.
        tgt_mask: mask Tensor on the decoder input.
        xe_mask: mask Tensor on the encoder output.
        optimizer: optimizer of the training.
        criterion: loss function to minimize.
        lr: learning rate value.

    """

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = train_data[0].size(0)
    for batch_nb in range(num_batches):
        
        x, y_input, gender_index, y_expected  = get_batch(train_data, batch_nb)
        seq_len = x.size(1)

        if seq_len != batch_size:  # only on last batch
            xe_mask = xe_mask[:seq_len, :seq_len]
        output = model(x, y_input, gender_index, xe_mask, tgt_mask)
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

            print(f'| epoch {epoch:3d} | {batch_nb:5d}/{num_batches:5d} batches | lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

            total_loss = 0
            start_time = time.time()



def evaluate(
    model: torch.nn.Module, 
    batch_size: int,
    data: Tensor,  
    tgt_mask: Tensor,   
    xe_mask: Tensor, 
    criterion = torch.nn.MSELoss) -> float:

    """Calculates the model loss for the given data.

    Args:
        model: model to train.
        data: Tensor with data in the format (xe_test, xd_test, gender_idx_test, yd_test)
        xe_mask: mask Tensor on the encoder output.
        tgt_mask: mask Tensor on the decoder input.
    
    Returns:
        float with loss value.
    
    """

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    num_batches = data[0].size(0)
    batch_size = data[0].size(1)
    num_val_patterns = num_batches * batch_size

    with torch.no_grad():
        for batch_nb in range(num_batches):
            x_val, y_val_input, gender_idx, y_val_expected = get_batch(data, batch_nb)
            seq_len = x_val.size(1)
            if seq_len != bptt:
                xe_mask = xe_mask[:seq_len, :seq_len]
            output = model(x_val, y_val_input, gender_idx , xe_mask, tgt_mask)
            total_loss += seq_len * criterion(output, y_val_expected).item()

    return total_loss /  (num_val_patterns - 1)


def fit(
    model: torch.nn.Module, 
    batch_size: int,
    epochs: int, 
    train_data: Tensor, 
    val_data: Tensor, 
    xe_mask: Tensor, 
    tgt_mask: Tensor, 
    opt: torch.optim, 
    criterion = torch.nn.MSELoss, 
    lr: float = 0.05,
    scheduler: torch.optim.lr_scheduler = None) -> torch.nn.Module:

    """Fits the transformer model to the training data.

    Args:
        model: model to train.
        batch_size: size of each batch.
        epochs: maximum number of epochs to train.
        train_data: data for training.
        val_data: data for validation.
        xe_mask: mask Tensor on the encoder output.
        tgt_mask: mask Tensor on the decoder input.
        opt: optimizer of the training.
        criterion: loss function to minimize.
        lr: learning rate value.
        scheduler: function responsable to vary the learning rate.

    Returns:
        Trained model of the class torch.nn.Module with the best model according to the validation data.
    
    """
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train(
        model = model, 
        batch_size = batch_size,
        train_data = train_data,
        xe_mask = xe_mask,
        tgt_mask = tgt_mask,
        epoch = epoch, 
        optimizer = opt, 
        lr = lr,
        criterion = criterion)

        val_loss = evaluate(model,batch_size, val_data, tgt_mask, xe_mask, criterion)

        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if scheduler is not None:
            scheduler.step()
        
    return best_model

