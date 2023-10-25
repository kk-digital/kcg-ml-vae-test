import torch
import tqdm
import os
import time

import numpy as np
import matplotlib.pyplot as plt

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LossTracker:
    def __init__(self):
        self.losses = {}  # Initialize an empty dictionary to store the losses
        self.counts = {}  # Initialize a dictionary to keep track of the counts for each loss key

    def update(self, loss_dict, batch_size):
        """
        Update the loss tracker with a dictionary of losses.

        Args:
            loss_dict (dict): A dictionary containing loss values with keys.
        """
        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key] += (value / batch_size)
                self.counts[key] += 1
            else:
                self.losses[key] = (value / batch_size)
                self.counts[key] = 1

    def average(self):
        """
        Calculate the average of tracked losses.

        Returns:
            dict: A dictionary containing the average losses for each key.
        """
        averaged_losses = {}
        for key in self.losses:
            averaged_losses[key] = self.losses[key] / self.counts[key]
        return averaged_losses

    def reset(self):
        """
        Reset the loss tracker, clearing all tracked losses and counts.
        """
        self.losses = {}
        self.counts = {}

def simple_train_step(dataloader, model, device, criterion, optimizer, clip_value=1.0):
    model.train()

    losses = []
    pbar = tqdm.tqdm(dataloader)
    loss_tracker = LossTracker()
    for data in pbar:
        x = data[0]
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        y = model(x)

        loss, loss_dict = criterion(y, x, model.z, model)

        # Backward pass
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        avg_loss = loss / x.shape[0]
        optimizer.step()
        losses.append(avg_loss)
        loss_tracker.update(loss_dict, x.shape[0])
        pbar.set_description(f'train loss: {avg_loss.item():.3f}')

    all_loss = loss_tracker.average()
    loss_tracker.reset()
    losses = torch.stack(losses, 0)
    epoch_loss = torch.mean(losses).cpu()

    return epoch_loss, all_loss, model, optimizer

def simple_eval_step(dataloader, model, criterion, device):
    model.eval()
    pbar = tqdm.tqdm(dataloader)
    losses = []
    loss_tracker = LossTracker()
    for data in pbar:
        x = data[0]
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
            loss, loss_dict = criterion(y, x, model.z, model)
            avg_loss = loss / x.shape[0]
            losses.append(avg_loss)
        pbar.set_description(f'val loss: {avg_loss.item():.3f}')
        loss_tracker.update(loss_dict, x.shape[0])

    all_loss = loss_tracker.average()
    losses = torch.stack(losses, 0)
    final_loss = torch.mean(losses).cpu()
    
    return final_loss, all_loss

def train_loop(
    train_loader,
    val_loader,
    model,
    device,
    criterion, 
    optimizer,
    scheduler,
    epochs,
    save_path,
    eval_every=1,
    clip_value=1.0
):
    os.makedirs(os.path.join(save_path, 'weights'), exist_ok=True)
    best_loss = 1e+15
    train_losses = []
    val_losses = []
    learning_rates = []
    start_time = time.time()
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        train_loss, train_all_loss, model, optimizer = simple_train_step(
            train_loader, model, device, criterion, optimizer, clip_value)
        train_losses.append(train_loss.detach().numpy())

        if (epoch+1)%eval_every == 0:
            val_loss, val_all_loss = simple_eval_step(val_loader, model, criterion, device)
            val_losses.append(val_loss.cpu().numpy())

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            print(f'Weight saved: epoch {epoch}')
            torch.save(model, os.path.join(save_path, 'weights', 'best.pt'))
            best_loss = val_loss

        print(f'Epoch {epoch}\tLR: {current_lr:.6f}')
        train_loss_str = 'Train Losses: '
        for k, v in train_all_loss.items():
            train_loss_str += f'{k}: {v:.3f}\t'
        val_loss_str = 'Val Losses: '
        for k, v in val_all_loss.items():
            val_loss_str += f'{k}: {v:.3f}\t'
        print(f'{train_loss_str}\n{val_loss_str}\n')
        if (epoch + 1) % 10 == 0:
            training_time = time.time()
            print(f'Training time: {(training_time-start_time):.2f} seconds')
        
    torch.save(model, os.path.join(save_path, 'weights', 'last.pt'))

    fig, ax = plt.subplots()
    ax.plot(range(epochs), train_losses, label='Training Loss')
    ax.plot(range(0, epochs, eval_every), val_losses, label='Validation Loss')
    ax.set(xlabel='Epoch', ylabel='Loss')
    ax.legend()

    fig.savefig(os.path.join(save_path, 'loss.jpg'))

    fig_lr, ax_lr = plt.subplots()
    ax_lr.plot(range(epochs), learning_rates, label='Learning Rate')
    ax_lr.set(xlabel='Epoch', ylabel='Learning Rate')
    ax_lr.legend()

    fig_lr.savefig(os.path.join(save_path, 'learning_rate.jpg'))

    mean_val_loss = np.mean(val_losses)
    mean_train_loss = np.mean(train_losses)

    training_time = time.time()
    print(f'Total training time: {(training_time-start_time):.2f} seconds')

    return fig, mean_train_loss, mean_val_loss, best_loss

