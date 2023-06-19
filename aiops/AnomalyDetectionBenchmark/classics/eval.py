import torch
from torch import nn

@torch.no_grad()
def eval_LSTM(test_loader, model):
    y_pred = []
    model = model.cuda()
    criterion = nn.MSELoss(reduction='none')
    for batch in test_loader:
        X, X_next = batch
        pred = model(X.cuda())
        score = criterion(pred, X_next.cuda())
        score = torch.sum(score, dim=1)

        y_pred.extend(score.cpu().tolist())
        #y_true.extend(y.cpu().tolist())

    return y_pred

@torch.no_grad()
def eval_LSTM_AE(test_loader, model):
    y_pred = []
    model = model.cuda()
    criterion = nn.MSELoss(reduction='none')
    for batch in test_loader:
        X = batch
        X_hat = model(X.cuda())
        score = criterion(X.cuda(), X_hat)
        score = torch.sum(score.reshape(score.size(0), -1), dim=1)

        y_pred.extend(score.cpu().tolist())

    return y_pred


@torch.no_grad()
def eval_LSTM_VAE(test_loader, model):
    y_pred = []
    model = model.cuda()
    criterion = nn.MSELoss(reduction='none')
    for batch in test_loader:
        X = batch
        X_hat = model(X.cuda())
        score = criterion(X.cuda(), X_hat)
        score = torch.sum(score.reshape(score.size(0), -1), dim=1)

        y_pred.extend(score.cpu().tolist())

    return y_pred
