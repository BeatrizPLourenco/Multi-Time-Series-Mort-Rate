
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
import shutil
import matplotlib.pyplot as plt

def save_plots(train_loss: list, valid_loss: list, gender: str) -> None:
    """
    Function to save the loss and accuracy plots to disk.
    """
    plt.rcParams.update({'font.size': 18})

    # loss plots
    plt.figure(figsize=(10, 10))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validation loss'
    )
    plt.xlabel('Epochs',fontsize = 18, fontweight='bold')
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.ylabel('Loss',fontsize = 18, fontweight='bold')
    plt.legend(prop=dict(size=18))
    plt.savefig(f'Images/loss{gender}.pdf')


def save_ckp(state: dict, is_best: bool, checkpoint_dir: str = 'Saved_models/checkpoint.pt', best_model_dir: str = 'Saved_models/best_model.pt') -> None:
    """Saves the current state of training into a directory.

    Args:
        state: dictionary with all the desired information to save.
        is_best: True if the model saved is the best model (lowest validation loss) so far. False otherwise
        checkpoint_dir: directory to save the checkpoint.
        best_model_dir: directory to save the best model.
    """

    f_path = checkpoint_dir
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir
        shutil.copyfile(f_path, best_fpath)


def load_ckp( model, optimizer, scheduler, checkpoint_dir: str = 'Saved_models/checkpoint.pt') -> None:
    """ Loads the latest checkpoint into the model and the optimizer.

    Args: 
        model: nn.Module to substitute the current state.
        optimizer: optimizer to substitute the current state.
        checkpoint_dir: directory to save the checkpoint."""
    

    checkpoint_fpath = checkpoint_dir 
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    checkpoint_history = checkpoint['history']
    epoch = checkpoint['epoch']

    return epoch, model, optimizer, scheduler, checkpoint_history




def load_best_model(model, best_model_dir: str = 'Saved_models/best_model.pt'):
    """Loads the best model from the disk.

    Args:
        best_model_dir: folder location of the best model. The file name most be '/best_model.pt'.

    Returns:
        model: model with uploaded weigths.
        checkpoint_history: dictonary with training_loss_history, val_loss_history, lr_history
    """

    checkpoint = torch.load(best_model_dir)
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint_history = checkpoint['history']

    return model, checkpoint_history


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(source: tuple, i: int) -> tuple:
    """Gets specific batch in a set.

    Returns:
        Tensor with desired batch.
    """
    
    return source[0][i, :, :, :], source[1][i, :, :, :], source[2][i, :] if source[2] is not None else None, source[3][i, :]



def train(
    model: torch.nn.Module,
    batch_size: int,
    epoch: int,
    train_data: Tensor,
    tgt_mask: Tensor,
    xe_mask: Tensor,  
    optimizer: torch.optim, 
    criterion = torch.nn.MSELoss,
    scheduler: torch.optim.lr_scheduler = None,) -> None:

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

        if scheduler is not None:
            last_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

        total_loss += loss.item()
        if batch_nb % log_interval == 0 and batch_nb > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch_nb:5d}/{num_batches:5d} batches | lr {last_lr:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f}')

            total_loss = 0
            start_time = time.time()
    
    return {'loss' : cur_loss, 'lr': last_lr}



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
            if seq_len != batch_size:
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
    scheduler: torch.optim.lr_scheduler = None,
    resume_training: bool = False,
    patience = 50,
    checkpoint_dir:str = 'Saved_models/checkpoint.pt',
    best_model_dir: str = 'Saved_models/best_model.pt'):

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
        resume_training: If true, resumes training from the latest checkpoint. Restarts training, otherwise.

    Returns:
        Trained model of the class torch.nn.Module with the best model according to the validation data.
    
    """
    is_best = False
    start_epoch = 1
    lr_history = []
    train_loss_history = [float('inf')]
    val_loss_history = [float('inf')]

    if resume_training:

        start_epoch, model, opt, scheduler, checkpoint_history = load_ckp( model, opt, scheduler, checkpoint_dir )
        train_loss_history, val_loss_history, lr_history = checkpoint_history['train_loss_history'], checkpoint_history['val_loss_history'], checkpoint_history['lr_history']

    best_model = model
    lowest_val_loss = val_loss_history[-1]

    epochs_without_improvement = 0
    for epoch in range(start_epoch, epochs + 1):
        is_best = False
        
        epoch_start_time = time.time()

        cur_train_loop = train(
            model = model, 
            batch_size = batch_size,
            train_data = train_data,
            xe_mask = xe_mask,
            tgt_mask = tgt_mask,
            epoch = epoch, 
            optimizer = opt, 
            criterion = criterion,
            scheduler = scheduler)
        train_loss_history.append(cur_train_loop['loss'])
        lr_history.append(cur_train_loop['lr'])

        val_loss = evaluate(model,batch_size, val_data, tgt_mask, xe_mask, criterion)


        if val_loss < lowest_val_loss:
            is_best = True
            lowest_val_loss = val_loss
            best_model = model
            epochs_without_improvement = 0
        
        else:
          epochs_without_improvement +=1

        if epochs_without_improvement > patience:
          break

        
            
        val_loss_history.append(val_loss)

        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f}')
        print('-' * 89)


        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': {'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'lr_history': lr_history}
            }

        save_ckp(checkpoint, is_best,checkpoint_dir = checkpoint_dir, best_model_dir = best_model_dir )

    
    history = {
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'lr_history': lr_history
        }
    return best_model, history

