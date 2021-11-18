import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import copy
from model_utils import save_ckp
import pdb

# from https://github.com/anudeepsekhar/deep_learning_for_demosaicing/blob/6ef83c2763ce06ae2df18c4ac7fb1c26881f352d/training.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, optimizer, scheduler, num_epochs,start_epoch,checkpoint_dir,dataloaders,clip_norm=None,scheduler_no_arg=None):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in (range(start_epoch,num_epochs)):
        print('\n Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # train
        model.train()
        print("Training...")
        training_loss = 0.0

        for inputs, targets in tqdm(dataloaders['train']):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            result = model(inputs)
            # print('result.shape: ',result.shape)

            # Calculate loss
            # pdb.set_trace()
            criterion = nn.MSELoss()
            loss = criterion(result.float(), targets.float())

            loss.backward()

            # clip gradient if applicable
            if clip_norm is not None:
              nn.utils.clip_grad_norm(model.parameters(),clip_norm)

            optimizer.step()

            training_loss += loss.item()

        num_batch = len(dataloaders['train'])
        training_loss /= num_batch

        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }
        # saves the checkpoint for resuming in case instance disconnected

        save_ckp(checkpoint, checkpoint_dir)

        print('Epoch: {}, Average training loss: {:.6f}'.format(epoch+1,training_loss))

        # validate
        print("Validating...")
        model.eval()
        val_loss = 0.0

        for inputs, targets in tqdm(dataloaders['val']):
          with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)

            result = model(inputs)

            # Calculate loss
            criterion = nn.MSELoss()
            loss = criterion(result.float(), targets.float())

            val_loss += loss.item()
            # statistics

        num_batch = len(dataloaders['val'])
        val_loss /= num_batch

        print('Epoch: {}, Average validating loss: {:.6f}'.format(epoch+1,val_loss))
        if scheduler_no_arg is None:
          scheduler.step(val_loss)
        elif scheduler_no_arg is True:
          scheduler.step()


        # deep copy the model
        if val_loss < best_loss:
            print("saving best model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:6f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
