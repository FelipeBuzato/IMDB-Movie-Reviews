import pandas as pd

class DataLoader:
    def __init__(self, path_train, path_test):
        self.path_train = path_train
        self.path_test = path_test


    def load(self):
        train_data = pd.read_csv(self.path_train)
        test_data = pd.read_csv(self.path_test)
        return train_data, test_data