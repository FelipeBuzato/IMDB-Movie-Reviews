from sklearn.linear_model import LogisticRegression

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
        
        raise ValueError(f"Unknown model: {model_name}")