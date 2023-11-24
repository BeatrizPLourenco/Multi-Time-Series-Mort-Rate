import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from train_transformer import save_ckp, load_ckp
import time

def get_batch(source: tuple, i: int, batch_size: int) -> tuple:
    """Gets specific batch in a set.

    Returns:
        Tensor with desired batch.
    """
    # Extract the current batch from the training data and targets
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    input_batch = source[0][start_idx:end_idx,:, :]
    gender_batch = source[1][start_idx:end_idx] if source[1] is not None else None
    target_batch = source[2][start_idx:end_idx]



    return input_batch, gender_batch, target_batch
    #return source[0][i, :, :, :], source[1][i, :] if source[1] is not None else None, source[2][i, :]



def train(
    model: torch.nn.Module,
    batch_size: int,
    epoch: int,
    train_data: Tensor,
    optimizer: torch.optim, 
    criterion = torch.nn.MSELoss,
    scheduler: torch.optim.lr_scheduler = None,) -> None:

    """Trains the model through 1 epoch.

    Args:
        model: model to train.
        batch_size: size of each batch.
        epoch: epoch number.
        train_data: data for training.
        optimizer: optimizer of the training.
        criterion: loss function to minimize.
        lr: learning rate value.

    """

    model.train()  # turn on train mode
    total_loss = 0.
    loss_log_interval = 0.
    log_interval = 50
    start_time = time.time()

    num_batches = train_data[0].size(0) // batch_size
    for batch_nb in range(num_batches):
        
        x, gender_index, y_expected  = get_batch(train_data, batch_nb, batch_size)
        seq_len = x.size(1)

        output = model(x, gender_index)
        loss = criterion(output, y_expected)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        last_lr = 0
        if scheduler is not None:
            last_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

        total_loss += loss.item()
        loss_log_interval += loss.item()
        if batch_nb % log_interval == 0 and batch_nb > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = loss_log_interval /  (log_interval - 1)
            print(f'| epoch {epoch:3d} | {batch_nb:5d}/{num_batches:5d} batches | lr {last_lr:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                f'log interval loss {cur_loss:5.2f}')

            loss_log_interval = 0
            start_time = time.time()
        
    loss_per_batch =  total_loss / (num_batches - 1)
    
    return {'loss' : loss_per_batch, 'lr': last_lr}



def evaluate(
    model: torch.nn.Module, 
    batch_size: int,
    data: Tensor, 
    criterion = torch.nn.MSELoss) -> float:

    """Calculates the model loss for the given data.

    Args:
        model: model to train.
        data: Tensor with data in the format (xe_test, xd_test, gender_idx_test, yd_test)
    
    Returns:
        float with loss value.
    
    """

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    num_samples = 0.
    num_val_patterns = data[0].size(0)
    num_val_batches = num_val_patterns // batch_size

    with torch.no_grad():
        for batch in range(num_val_batches):
            x_val, gender_idx, y_val_expected = get_batch(data, batch, batch_size)
            output = model(x_val, gender_idx)
            num_samples += batch_size
            total_loss += criterion(output, y_val_expected).item() * batch_size


    return total_loss / num_samples


def fit(
    model: torch.nn.Module, 
    best_model: torch.nn.Module,
    batch_size: int,
    epochs: int, 
    train_data: Tensor, 
    val_data: Tensor, 
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
    num_batches_val = val_data[0].size(0) // batch_size
    start_epoch = 1
    lr_history = []
    train_loss_history = [float('inf')]
    val_loss_history = [float('inf')]

    if resume_training:

        start_epoch, model, opt, scheduler, checkpoint_history = load_ckp( model, opt, scheduler, checkpoint_dir )
        train_loss_history, val_loss_history, lr_history = checkpoint_history['train_loss_history'], checkpoint_history['val_loss_history'], checkpoint_history['lr_history']

    best_model.load_state_dict(model.state_dict())
    lowest_val_loss = val_loss_history[-1]

    epochs_without_improvement = 0
    for epoch in range(start_epoch, epochs + 1):
        is_best = False
        
        epoch_start_time = time.time()

        cur_train_loop = train(
            model = model, 
            batch_size = batch_size,
            train_data = train_data,
            epoch = epoch, 
            optimizer = opt, 
            criterion = criterion,
            scheduler = scheduler)
        train_loss = cur_train_loop['loss']
        train_loss_history.append(train_loss)
        lr_history.append(cur_train_loop['lr'])

        val_loss = evaluate(model,batch_size, val_data,criterion)

        print(f'*** lowest_val_loss = {lowest_val_loss} ***')
        if val_loss < lowest_val_loss:
            is_best = True
            lowest_val_loss = val_loss
            best_model.load_state_dict(model.state_dict())
            
            epochs_without_improvement = 0
        
        else:
          epochs_without_improvement +=1

        if epochs_without_improvement > patience:
          break

        
            
        val_loss_history.append(val_loss)

        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | train loss {train_loss:5.2f}')
        print('-' * 89)


        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
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
    print(f'Best model evaluation: {evaluate(best_model,batch_size, val_data,criterion)}')
    return best_model, history

