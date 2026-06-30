import re
from bs4 import BeautifulSoup
import html

class PreProcessorDL:
    def __init__(self, lowercase=True, remove_punctuation=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.target = 'sentiment'
        self.cols_to_drop = ['id']


    def _clean_text(self, text):
        # put everything in lowercase
        if(self.lowercase): text = text.lower()

        # Decode HTML entities
        text = html.unescape(text)

         # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text(" ")

        # Remove escape characters
        text = re.sub(r"[\n\t\r]+", " ", text)

        # remove punctuation (keep apostrophes)
        if(self.remove_punctuation): text = re.sub(r"[^\w\s']", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        X = X.copy()
        columns = X.columns.tolist()
        
        # remove unnecessary columns
        cols_to_drop = self.cols_to_drop.copy()
        
        if(self.target in columns):
            cols_to_drop.append(self.target)
                
        X = X.drop(columns=cols_to_drop)

        # clean text in reviews
        for col in X.columns.tolist():
            X[col] = X[col].apply(self._clean_text)

        return X 


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)