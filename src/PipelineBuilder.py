from PreProcessorClassicML import PreProcessorClassicML
from PreProcessorDeep import PreProcessorDeep
from ModelCollection import ModelCollection
from sklearn.pipeline import Pipeline

class PipelineBuilder:
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
            preprocessor = self.get_preprocessor_classic(vectorizer=vectorizer)
        
        elif(model_name in ["RNN"]):
            preprocessor = self.get_preprocessor_deep()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        return pipeline
    

    def get_preprocessor_classic(self, vectorizer="TF-IDF"):
        return PreProcessorClassicML(self.data).build(vectorizer_type=vectorizer)


    def get_preprocessor_deep(self):
        return PreProcessorDeep(self.data).build()