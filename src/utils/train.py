import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model(model, train_loader, epoch, max_epoch, device, optimizer, n_iter):
    model.train()
    correct = 0
    total = 0
    for (fonts, labels) in tqdm(
        train_loader,
        desc=f'Train {epoch}/{max_epoch - 1}',
        total=len(train_loader),
    ):
        fonts = fonts.to(device)
        labels = labels.to(device)

        outputs = model(fonts)
        loss = F.cross_entropy(outputs, labels, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy = (outputs.max(1)[1] == labels).float().mean().item()
        correct += (outputs.max(1)[1] == labels).sum().item()
        total += len(labels)
        n_iter += 1

    print(f'train loss : {loss.item()} train accuracy : {correct/total}')

    return n_iter


def eval_model(model, val_loader, epoch, max_epoch, device, n_iter):
    model.eval()

    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (fonts, labels) in tqdm(
            val_loader,
            desc=f'Test {epoch}/{max_epoch - 1}',
            total=len(val_loader),
        ):
            fonts = fonts.to(device)
            labels = labels.to(device)

            outputs = model(fonts)

            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.max(1)[1] == labels).sum().item()
            total += len(labels)

        loss /= total
        accuracy = correct / total
    print(f'validation loss : {loss} validation accuracy : {accuracy}')

    return loss
