from pydantic import BaseModel

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    incidenceRate: float
    povertyPercent: float
    MedianAgeFemale: float
    PctHS18_24: float
    PctHS25_Over: float
    PctUnemployed16_Over: float
    PctPrivateCoverage: float

#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ['incidenceRate', 'povertyPercent', 'MedianAgeFemale',
                    'PctHS18_24', 'PctHS25_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage']
        

