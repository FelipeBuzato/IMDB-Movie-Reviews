from Trainer import Trainer
from PreProcessorDL import PreProcessorDL
from Vectorizer import Vectorizer
from Vectorizer import Word2VecEmbedding
from ModelCollection import ModelCollection
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class PipelineDL(BaseEstimator):
    def __init__(self, preprocessor_type="default", lowercase=None, remove_punctuation=None,
                 vectorizer_method="word2vec", embedding_dim=None, window=None, min_count=None, max_length=None, min_df=None, n_gram_range=None,
                 model_name="RNN", input_dim=None, hidden_dim=None, output_dim=None, num_layers=None, linear_layer_sizes=None,
                 optimizer=None, lr=None, epochs=None, batch_size=None, criterion=None, random_state=42):
        
        # preprocessor hyper-parameters
        self.preprocessor_type = preprocessor_type
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

        # vectorizer hyper-parameters
        self.vectorizer_method = vectorizer_method
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.max_length = max_length
        self.min_df = min_df
        self.n_gram_range = n_gram_range

        # model hyper-parameters
        self.model_name = model_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.linear_layer_sizes = linear_layer_sizes

        # trainer hyper-parameters
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion

        self.random_state = random_state

        self.preprocessor_ = None
        self.vectorizer_ = None
        self.model_collection_ = None
        self.model_ = None
        self.trainer_ = None
        

    def fit(self, X, y):
        X = X.copy()
        self.preprocessor_ = self.get_preprocessor()
        self.vectorizer_ = self.get_vectorizer()
        self.model_collection_ = ModelCollection()
        model_params = self.get_model_params(self.model_name)
        self.model_ = self.model_collection_.get(self.model_name, model_params)

        X = self.preprocessor_.fit_transform(X)
        X = self.vectorizer_.fit_transform(X)

        trainer_params = self._get_subset_params(["optimizer", "lr", "epochs", "batch_size", "criterion", "random_state"])
        self.trainer_ = Trainer(**trainer_params)
        self.trainer_.fit(self.model_, X, y)

        return self


    def predict(self, X):
        X = X.copy()
        X = self.preprocessor_.transform(X)
        X = self.vectorizer_.transform(X)
        return self.trainer_.predict(self.model_, X)


    def predict_proba(self, X):
        X = X.copy()
        X = self.preprocessor_.transform(X)
        X = self.vectorizer_.transform(X)
        return self.trainer_.predict_proba(self.model_, X)


    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


    def reset(self):
        self.model_collection_ = None
        self.preprocessor_ = None
        self.vectorizer_ = None
        self.model_ = None
        self.trainer_ = None

    
    def _get_subset_params(self, keys):
        # get all params that aren't None
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
    

    def get_model_params(self, model_name):
        if(model_name == "RNN"):
            keys = ["input_dim", "hidden_dim", "output_dim", "num_layers"]
        elif(model_name == "MLP"):
            keys = ["linear_layer_sizes"]
        else:
            raise ValueError(f"Model name {model_name} not found.")
        
        return self._get_subset_params(keys)


    def get_preprocessor(self):
        if(self.preprocessor_type == "default"):
            keys = ["lowercase", "remove_punctuation"]
            preprocessor_params = self._get_subset_params(keys=keys)
            return PreProcessorDL(**preprocessor_params)
        
        raise ValueError(f"Preprocessor type {self.preprocessor_type} not found.")
        
    
    def get_vectorizer(self):
        if(self.vectorizer_method == "word2vec"):
            keys = ["embedding_dim", "window", "min_count", "max_length", "random_state"]
            embedding_params = self._get_subset_params(keys=keys)
            return Word2VecEmbedding(**embedding_params)
        
        elif(self.vectorizer_method in ["bag of words", "tf-idf"]):
            if(self.model_name != "MLP"): 
                raise ValueError(f"{self.vectorizer_method} can only be used with non-sequencial NNs. Change model name.")
            keys = ["vectorizer_method", "min_df", "n_gram_range"]
            vectorizer_params = self._get_subset_params(keys=keys)
            return Vectorizer(**vectorizer_params)
        
        raise ValueError(f"Vectorizer method {self.vectorizer_method} not found.")
    

    @property
    def named_steps(self):
        return {
            "preprocessor": self.preprocessor_,
            "vectorizer": self.vectorizer_,
            "model": self.model_,
        }