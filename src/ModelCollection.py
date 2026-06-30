from sklearn.linear_model import LogisticRegression
import torch.nn as nn

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
        
        raise ValueError(f"Unknown model: {model_name}")
    

class RNN(nn.Module):
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, num_layers=1):
        super().__init__()


    def forward(self, X):
        pass


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