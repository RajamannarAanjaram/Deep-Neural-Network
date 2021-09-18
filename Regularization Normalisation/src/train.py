import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train(model, device, train_loader, optimizer, train_acc, train_losses, lambda_l1,):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    loss_rec = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        if lambda_l1:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss += lambda_l1 * l1

        # Backpropagation
        loss.backward()
        loss_rec = loss_rec + loss
        optimizer.step()

        # Update pbar-tqdm

        # get the index of the max log-probability
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_losses.append(loss_rec/len(train_loader.dataset))
    train_acc.append(100*correct/processed)
