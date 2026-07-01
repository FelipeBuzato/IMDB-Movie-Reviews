from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

DEFAULT_PARAMS = {
    "Logistic Regression": {'l1_ratio': 0, 'solver': 'saga', 'C': 1.0, 'max_iter': 1000}, 
}

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params=None):
        if params is None:
            if(model_name not in DEFAULT_PARAMS):
                params = {}
            else: params = DEFAULT_PARAMS[model_name] 

        if(model_name == "Logistic Regression"):
            return LogisticRegression(**params)
        
        elif(model_name == "MLP"):
            return MLP(**params)
        
        elif(model_name == "RNN"):
            return RNN(**params)
        
        elif(model_name == "LSTM"):
            return LSTM(**params)
        
        raise ValueError(f"Unknown model: {model_name}")
    

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, fine_tunning=False, padding_idx=0,
                 num_layers=1, hidden_dim=128, output_dim=2, dropout=0.0, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        if(embedding_matrix is not None):
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_matrix)

            if(fine_tunning ==  False):
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        multiplier = 2 if bidirectional else 1

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(hidden_dim * multiplier, output_dim)


    def forward(self, X):
        lengths = (X != 0).sum(dim=1)
        embedded = self.embedding(X)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(packed)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return output
    

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, fine_tunning=False, padding_idx=0, 
                 num_layers=1, hidden_dim=128, output_dim=2, dropout=0.0, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        if(embedding_matrix is not None):
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_matrix)

            if(fine_tunning == False):
                self.embedding.weight.requires_grad = False

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        multiplier = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden_dim * multiplier, output_dim)


    def forward(self, X):
        lengths = (X != 0).sum(dim=1)
        embedded = self.embedding(X)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, hidden = self.rnn(packed)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        output = self.fc(hidden)
        return output


class MLP(nn.Module):
    def __init__(self, linear_layer_sizes=(128,), output_size=2):
        super().__init__()
        
        sizes = list(linear_layer_sizes) + [output_size]
        layers = []
        layers.append(nn.LazyLinear(linear_layer_sizes[0]))
        layers.append(nn.ReLU())

        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)


    def forward(self, X):
        return self.network(X)