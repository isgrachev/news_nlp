import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# os.chdir('..')
# print(os.getcwd())

import torch
from torch import nn
# from torch.utils.data import TensorDataset, DataLoader

import gensim


class LSTMClassifier(nn.Module):
    """
    
    """

    def __init__(self, num_embeddings=5084, embedding_dim=64, num_layers=1, hidden_size=32, out_features=5):
        super().__init__()
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim).from_pretrained(embedder)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=out_features)
        self.softmax = nn.Softmax()

    def forward(self, X):
        embed = self.embedding(X)
        rnn_output, hidden = self.lstm(embed)
        self.hidden = hidden
        self.rnn_output = rnn_output
        fc1 = self.fc1(hidden[0][1, :, :])
        # fc1 = self.fc1(rnn_output.mean(1))
        a1 = self.act1(fc1)
        fc2 = self.fc2(a1) # 
        out = self.softmax(fc2)
        return out


def fit(lstm, optimizer, criterion, loaders, epochs, device):
    """
    
    """

    train_logger = []
    val_logger = []

    for epoch in range(epochs):
        epoch_val_loss = []
        epoch_train_loss = []
        for k, dataloader in loaders.items():
            for doc_batch, label_batch in dataloader:
                doc_batch = doc_batch.to(device)
                label_batch = label_batch.to(device)
                
                if k == 'train':
                    lstm.train()
                    optimizer.zero_grad()
                    outp = lstm.forward(doc_batch)

                else:
                    lstm.eval()
                    with torch.no_grad():
                        outp = lstm.forward(doc_batch)

                # pred = outp.argmax(-1)
                loss = criterion(outp, label_batch.flatten())

                if k == 'train':
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(loss.item())
                else:
                    epoch_val_loss.append(loss.item())
        train_logger.append(np.mean(epoch_train_loss))
        val_logger.append(np.mean(epoch_val_loss))
        # if epoch % 1 == 0:
        print(f'Epoch: {epoch}    |   Train loss: {np.mean(epoch_train_loss)}    |    Validation loss: {np.mean(epoch_val_loss)}')    
    return (train_logger, val_logger)
