import torch


def train_model(model, criterion, optimizer, loader, device, batch_size, cumulation=1):

    epoch_loss = 0.0
    accuracy = 0

    model.train().to(device)

    for batch_idx in range(len(loader)):
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")

        X, y = loader.get_batch(batch_idx)
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        loss = criterion(output, y)
        loss.backward()
        epoch_loss += loss.item()

        _, preds = torch.max(output, 1)

        # Cumulated gradient
        if batch_idx % cumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        accuracy += torch.sum(preds == y.data)

    epoch_loss = epoch_loss / len(loader)
    train_acc = accuracy.double() / (len(loader) * batch_size)

    return epoch_loss, train_acc


def eval_model(model, criterion, loader, device, batch_size):
    eval_loss = 0
    eval_acc = 0

    model.eval().to(device)

    for batch_idx in range(len(loader)):
        X, y = loader.get_batch(batch_idx)
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            output = model(X)

            loss = criterion(output, y)
            eval_loss += loss.item()

            _, preds = torch.max(output, 1)
            eval_acc += torch.sum(preds == y.data)

    eval_loss = eval_loss / len(loader)
    eval_acc = eval_acc.double() / (len(loader) * batch_size)
    return eval_loss, eval_acc

