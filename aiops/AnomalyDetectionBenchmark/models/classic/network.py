import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, X):
        z, _ = self.encoder(X)
        pred = self.fc(z[:, -1, :])

        return pred

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_AE, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, X):
        z, _ = self.encoder(X)
        X_hat, _ = self.decoder(z)

        return X_hat

class LSTM_VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=10):
        super(LSTM_VAE, self).__init__()
        # encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # mu and log variance
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

        # decoder
        self.decoder = nn.LSTM(latent_size, input_size, batch_first=True)

    def sample(self, mu, std):
        z = mu + torch.randn_like(std) * std
        return z

    def forward(self, X):
        z, _ = self.encoder(X)
        mu, logvar = self.mu(z), self.logvar(z)
        std = torch.exp(logvar / 2)

        z = self.sample(mu, std)
        X_hat, _ = self.decoder(z)

        return X_hat


