from joblib import load

class Model:

    def __init__(self,columns):
        self.model = load("./Laboratorio-2-main/model/model.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
