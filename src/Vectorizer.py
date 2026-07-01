import os
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time

class Word2VecEmbedding:
    def __init__(self, embedding_dim=100, window=5, min_count=2, max_length=200, random_state=42):        
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.max_length = max_length
        self.random_state = random_state
        self.workers = os.cpu_count()

        self.vocabulary_ = None
        self.word2vec_ = None
        self.embedding_matrix_ = None


    def fit(self, X):
        X = X.copy()

        # Create list of sentences. Each sentence is a list of words
        start = time.time()
        sentences = []
        for col in X.columns.tolist():
            sentences.extend(X[col].str.split().tolist())
        print("time list of sentences:", time.time()-start)

        lengths = [len(sentence) for sentence in sentences]
        print(f"Mean   : {np.mean(lengths):.1f}")
        print(f"Median : {np.median(lengths):.1f}")
        print(f"90%    : {np.percentile(lengths, 90):.0f}")
        print(f"95%    : {np.percentile(lengths, 95):.0f}")
        print(f"99%    : {np.percentile(lengths, 99):.0f}")
        print(f"Max    : {max(lengths)}")

        # train word2vec
        start = time.time()
        self.word2vec_ = Word2Vec(
                              sentences=sentences,
                              vector_size=self.embedding_dim,
                              window=self.window,
                              min_count=self.min_count,
                              workers=self.workers,
                              epochs=20,
                              sg=0,    # 1 for skip-gram, 0 for continuous bag-of-words 
                              seed=self.random_state
                             )
        print("time word2vec:", time.time()-start)
        
        start = time.time()
        # create vocabulary with indexes -> get from word2vec
        vocabulary = {'PAD': 0, 'UNK': 1}
        for word in self.word2vec_.wv.key_to_index:
            vocabulary[word] = len(vocabulary)
        self.vocabulary_ = vocabulary
        print("time vocabulary:", time.time()-start)

        start = time.time()
        # add PAD and UNK embeddings to the embeddings matrix
        w2v_matrix = self.word2vec_.wv.vectors.copy()
        embedding_matrix = np.empty((w2v_matrix.shape[0] + 2, w2v_matrix.shape[1]), dtype=np.float32)
        # PAD
        embedding_matrix[0] = 0.0
        # UNK
        embedding_matrix[1] = np.random.normal(loc=0.0, scale=0.01, size=w2v_matrix.shape[1])
        # Word2Vec
        embedding_matrix[2:] = w2v_matrix
        self.embedding_matrix_ = torch.from_numpy(embedding_matrix).float()
        print("time embedding matrix:", time.time()-start)
        
        return self


    def transform(self, X):
        X = X.copy()
        
        start = time.time()
        # make list of reviews -> each review becomes a list of tokens (words) -> tokenization
        X = X['review'].apply(str.split).tolist()
        print("time list of sentences2:", time.time()-start)

        start = time.time()
        # replace each review by a list of indices of size max_length
        # fill the remaining values with 0 (PAD value)
        X_indices = np.full((len(X), self.max_length), fill_value=self.vocabulary_["PAD"], dtype=np.int64)

        for i, review in enumerate(X):
            indices = [
                self.vocabulary_.get(word, self.vocabulary_["UNK"])
                for word in review[:self.max_length]
            ]
            X_indices[i, :len(indices)] = indices
        
        # transform the output into a tensor (n_samples, max_length)
        X_tensor = torch.from_numpy(X_indices).long()
        print("time list of indices:", time.time()-start)

        return X_tensor


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

class Vectorizer:
    def __init__(self, vectorizer_method="tf-idf", min_df=1, n_gram_range=(1,1)):
        self.vectorizer_method = vectorizer_method
        self.min_df = min_df
        self.n_gram_range = n_gram_range

        self.vectorizer_ = None


    def _get_text(self, X):
        """
        Accepts either:
            - DataFrame with one or more text columns
            - Series
        Returns a single text string per sample (Series).
        """

        if(isinstance(X, pd.Series)):
            return X.astype(str)

        if(isinstance(X, pd.DataFrame)):
            cols = X.columns.tolist()
            if(len(cols) == 1):
                return X[cols[0]]
            return X.astype(str).agg(" ".join, axis=1)

        raise TypeError("X must be a pandas DataFrame or Series.")


    def fit(self, X, y=None):

        if(self.vectorizer_method == "tf-idf"):
            self.vectorizer_ = TfidfVectorizer(
                                                lowercase=False,
                                                token_pattern=r"(?u)\b[\w']+\b",
                                                ngram_range=self.n_gram_range,
                                                min_df=self.min_df,
                                                max_features=20000
                                              )

        elif(self.vectorizer_method == "bag of words"):
            self.vectorizer_ = CountVectorizer(
                                                lowercase=False,
                                                token_pattern=r"(?u)\b[\w']+\b",
                                                ngram_range=self.n_gram_range,
                                                min_df=self.min_df,
                                                max_features=20000
                                              )

        else:
            raise ValueError(f"Unknown vectorizer '{self.vectorizer_method}'.")

        texts = self._get_text(X)
        
        start = time.time()
        self.vectorizer_.fit(texts)
        #print("Time fit tfidf: ", time.time()-start)
        #print("len texts:", len(texts))
        #print("text stats:", texts.str.len().describe())
        #print("len vocabulary:", len(self.vectorizer_.vocabulary_))

        return self
    

    def transform(self, X):
        X = X.copy()

        start = time.time()
        
        texts = self._get_text(X)
        #print("Time get text: ", time.time()-start)
        
        start = time.time()
        X = self.vectorizer_.transform(texts)
        #print("Time transform vectorizer: ", time.time()-start)

        start = time.time()
        X = X.astype("float32")
        X = torch.tensor(X.toarray(), dtype=torch.float32)
        #print("Time tensorize: ", time.time()-start)
        
        return X
    

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)