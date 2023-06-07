from torch import nn
import numpy as np

def fit_LSTM(train_loader, model, optimizer, epochs):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss_batch = []
        for batch in train_loader:
            X, X_next = batch

            # clear gradient
            model.zero_grad()
            model = model.cuda()
            # loss forward
            pred = model(X.cuda())
            loss = criterion(pred, X_next.cuda())

            # loss backward
            loss.backward()
            loss_batch.append(loss.item())

            # gradient update
            optimizer.step()

        print(f'Epoch: {epoch}, loss: {np.mean(loss_batch)}')


def fit_LSTM_AE(train_loader, model, optimizer, epochs):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss_batch = []
        for batch in train_loader:
            X = batch

            # clear gradient
            model.zero_grad()
            
            model = model.cuda()
            # loss forward
            X_hat = model(X.cuda())
            loss = criterion(X.cuda(), X_hat)

            # loss backward
            loss.backward()
            loss_batch.append(loss.item())

            # gradient update
            optimizer.step()

        print(f'Epoch: {epoch}, loss: {np.mean(loss_batch)}')

def fit_LSTM_VAE(train_loader, model, optimizer, epochs):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss_batch = []
        for batch in train_loader:
            X = batch
            # clear gradient
            model.zero_grad()
            model = model.cuda()
            # loss forward
            X_hat = model(X.cuda())
            loss = criterion(X.cuda(), X_hat)

            # loss backward
            loss.backward()
            loss_batch.append(loss.item())

            # gradient update
            optimizer.step()

        print(f'Epoch: {epoch}, loss: {np.mean(loss_batch)}')