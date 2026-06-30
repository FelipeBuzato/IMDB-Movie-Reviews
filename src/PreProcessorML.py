from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from bs4 import BeautifulSoup
import html

class PreProcessorML():
    def __init__(self, data):
        self.data = data
        column_groups = self.split_columns()
        self.nlp_columns = column_groups[0]


    def split_columns(self):
        all_columns = self.data.columns.tolist()
        all_columns.remove('id')
        if('sentiment' in all_columns):
            all_columns.remove('sentiment')

        nlp_columns = all_columns
        return [nlp_columns]


    def clean_text(self, text):
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Decode HTML entities
        text = html.unescape(text)

        # Remove escape characters
        text = text.replace("\n", " ").replace("\t", " ")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    

    def clean(self, X):
        return X.squeeze().apply(self.clean_text)


    def build(self, vectorizer_type="TF-IDF"):
        # removes unwanted characters from each review
        data_cleaner = FunctionTransformer(self.clean)

        # vectorizes each review, transforming into a vector of numbers
        if(vectorizer_type == "TF-IDF"):
            vectorizer = TfidfVectorizer(
                                        lowercase=True, 
                                        token_pattern=r"(?u)\b[\w']+\b",  # Keep apostrophes inside words
                                        ngram_range=(1,2),
                                        #min_df=2
                                        )   
        
        elif(vectorizer_type == "bag of words"):
            vectorizer = CountVectorizer(
                                        lowercase=True,
                                        token_pattern=r"(?u)\b[\w']+\b",  # Keep apostrophes inside words
                                        ngram_range=(1,2),
                                        #min_df=2
                                        )
        
        else: raise ValueError(f"Preprocessing type {type} not found.")
    
        review_pipeline = Pipeline([
            ('cleaner', data_cleaner),
            ('vectorizer', vectorizer)
        ])

        column_transformer = ColumnTransformer([
            ('nlp', review_pipeline, self.nlp_columns)
        ])

        return column_transformer