import torch
import tqdm
import os

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simple_train_step(dataloader, model, device, criterion, optimizer, clip_value=1.0):
    model.train()

    losses = []
    pbar = tqdm.tqdm(dataloader)
    for data in pbar:
        x = data[0]
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        y = model(x)

        loss = criterion(y, x, model.z)

        # Backward pass
        loss.backward()

        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        avg_loss = loss / x.shape[0]
        optimizer.step()
        losses.append(avg_loss)
        pbar.set_description(f'train loss: {avg_loss.item():.3f}')

    losses = torch.stack(losses, 0)
    epoch_loss = torch.mean(losses).cpu()

    return epoch_loss, model, optimizer

def simple_eval_step(dataloader, model, criterion, device):
    model.eval()
    pbar = tqdm.tqdm(dataloader)
    losses = []
    for data in pbar:
        x = data[0]
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
            loss = criterion(y, x, model.z)
            avg_loss = loss / x.shape[0]
            losses.append(avg_loss)
        pbar.set_description(f'val loss: {avg_loss.item():.3f}')

    losses = torch.stack(losses, 0)
    final_loss = torch.mean(losses).cpu()
    
    return final_loss

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
    for epoch in range(epochs):
        train_loss, model, optimizer = simple_train_step(
            train_loader, model, device, criterion, optimizer, clip_value)

        if (epoch+1)%eval_every == 0:
            val_loss = simple_eval_step(val_loader, model, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            print(f'Weight saved: epoch {epoch}')
            torch.save(model, os.path.join(save_path, 'weights', 'best.pt'))
            best_loss = val_loss

        print(f'Epoch {epoch}\tTrain Loss: {train_loss:.3f}  Val Loss: {val_loss:.3f}\n')
        
    torch.save(model, os.path.join(save_path, 'weights', 'last.pt'))
