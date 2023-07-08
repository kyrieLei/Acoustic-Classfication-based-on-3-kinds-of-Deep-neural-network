from config import *
from util.func import normalize_std,apply_diff_freq
from util.aug import mixup,core_mixup,spec_augmenter
import torch
import torchaudio



device="cuda" if torch.cuda.is_available() else "cpu"
ComputeDeltas = torchaudio.transforms.ComputeDeltas(win_length= 5)

def train(dataloader, model, loss_fn, optimizer, t,scheduler):
    conf = config()
    size = len(dataloader.dataset)
    train_loss = 0
    n_train = 0
    correct = 0
    model.train()

    for batch, data in enumerate(dataloader):
        X=data[0]
        y=data[1]

        if conf.MIXUP:
            X, y = mixup(X, y)

        X = normalize_std(X)
        #X2 = ComputeDeltas(X)
        #X2 = normalize_std(X2)
        #X = torch.cat((X, X2), 1)

        if conf.SPEC_AUG:
            X = spec_augmenter(X)
        X, y = X.to(device), y.to(device)
        # Compute prediction error

        pred = model(X)

        #pred=torch.LongTensor(pred.detach().numpy())
        loss=loss_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(t + 1)
        train_loss += loss.item()

        outputs = model(X)
        _, prediction = torch.max(outputs, 1)

        correct += (prediction == y).sum().item()



        n_train += len(X)
        if batch % 500 == 0:
            loss_current, acc_current, current = train_loss / n_train, correct / n_train, batch * len(X)
            print(
                f"Train Epoch: {t + 1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")

    loss_current, acc_current = train_loss / n_train, correct / n_train
    return loss_current, acc_current


def val(dataloader, model, loss_fn,t):
    size = len(dataloader.dataset)
    val_loss = 0
    n_val = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred,y)

            val_loss += loss.item()
            inputs = normalize_std(X)

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct += (prediction == y).sum().item()
            #total_prediction += prediction.shape[0]

            #_, predicted = torch.max(pred[:, :-3].detach(), 1)
            #_, y_predicted = torch.max(y.detach(), 1)
            #correct += (predicted == y_predicted).sum().item()

            n_val += len(X)
            if batch % 500 == 0:
                loss_current, acc_current, current = val_loss / n_val, correct / n_val, batch * len(X)
                print(
                    f"Val Epoch: {t + 1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")

    loss_current, acc_current = val_loss / n_val, correct / n_val
    return loss_current, acc_current
