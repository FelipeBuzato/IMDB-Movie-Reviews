from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, target_column, test_size=0.2, random_state=42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
    
    
    def split(self, data):
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        return X_train, X_test, y_train, y_test