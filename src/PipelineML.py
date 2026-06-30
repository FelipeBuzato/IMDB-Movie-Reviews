from PreProcessorML import PreProcessorML
from ModelCollection import ModelCollection
from sklearn.pipeline import Pipeline

class PipelineML:
    def __init__(self, data):
        self.data = data
        self.model_colection = ModelCollection()

    
    def build(self, model_name, params=None):
        vectorizer = "TF-IDF"
        if('vectorizer' in params):
            vectorizer = params['vectorizer']
            del params['vectorizer']

        model = self.model_colection.get(model_name=model_name, params=params)

        if(model_name in ["Logistic Regression"]):
            preprocessor = PreProcessorML(self.data).build(vectorizer_type=vectorizer)
        else: 
            raise ValueError("Use the Deep Learning pipeline.")
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        return pipeline