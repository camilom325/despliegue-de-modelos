# Práctica - Despliegue de modelos de aprendizaje automático

## Objetivos

- Reforzar el conocimiento adquirido en la construcción de pipelines.
- Exportar un modelo de aprendizaje automático utilizando la librería Joblib.
- Construir una API REST para desplegar un modelo de aprendizaje automático que será utilizado mediante peticiones HTTP.

## Herramientas

- Librerías de Python para procesamiento de datos y construcción de modelos como: Pandas, Scikit-Learn y Joblib. 
- Frameworks para desarrollo de API REST en Python: [FastAPI](https://fastapi.tiangolo.com/), Flask o Django.
- IDE de Python para ciencia de datos: Jupyter Lab o Google Colab.
- IDE de Python para la construcción de aplicaciones web: Visual Studio Code, PyCharm, entre otros.
- Cliente para la realización de peticiones HTTP: [Postman](https://www.postman.com/).

## Enunciado

Esta práctica está dividida en dos partes:

1. La construcción de pipelines para la automatización de los procesos de experimentación e inferencia con modelos de Aprendizaje automático.
2. La creación de una API REST que habilita el consumo de modelos analíticos basados en aprendizaje automático, mediante solicitudes HTTP para realizar entrenamiento y predicción bajo demanda.

**Utilizando el modelo construido para el laboratorio 2, ajuste el pipeline contruido, si lo requiere, que permita encapsular los diferentes procesos requeridos para la preparación de datos, el entrenamiento del modelo y la predicción como proceso final del mismo.**

**Este pipeline debe ser exportado a un archivo binario utilizando la librería Joblib para posteriormente ser utilizado dentro de la API REST mediante peticiones HTTP. La API REST debe estar compuesta por dos endpoints:**

- **El primero debe recibir mediante el "body" una o más instancias de datos con la totalidad de las características requeridas para las que se desea realizar la predicción. Este endpoint debe devolver una lista con la misma cardinalidad de las instancias recibidas, y en el mismo orden, que contenga las predicciones realizadas por el modelo para cada instancia de datos recibida.**
- **El segundo endpoint, debe estar en capacidad de enviar una cantidad relevante de instancias de datos que servirán para realizar un proceso de re-entrenamiento del modelo, razón por la cual, además de enviar las características, también será requisito enviar la variable objetivo. Como respuesta, el endpoint deberá devolver algunas métricas de desempeño como pueden ser el RMSE, MAE y/o R^2. Además, tenga presente que con el proceso de re-entrenamiento del modelo, el archivo binario deberá reemplazarse para ser tenido en cuenta como nueva versión la próxima vez que se solicite realizar una predicción mediante el primer endpoint.** 

**Los datos tanto de la solicitud como de la respuesta deben estar en formato JSON y deben respetar el esquema del CSV original proporcionado para el laboratorio 2. Recuerde que es el pipeline el encargado de hacer todas las preparaciones requeridas previo a hacer los procesos de entrenamiento o predicción.**

## Construcción de Pipelines -- Rercordar este proceso

Para realizar esta sección se recomienda utilizar JupyterLab para la construcción del Pipeline y la exportación del modelo. 

 El objetivo de crear un [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)  es automatizar todos los pasos realizados sobre los datos. Desde que salen de su fuente hasta que son ingresados al modelo de aprendizaje automático. Para un problema clásico, estos pasos incluyen: la selección de características o columnas, la imputación de valores no existentes, la codificación de variables categóricas utilizando diferentes técnicas como Label Encoding o One Hot Encoding y el escalamiento de variables numéricas en caso de ser necesario. Sin embargo, note que para problemas como el procesamiento de textos los pasos necesarios son diferentes. Además, como último paso, el pipeline contiene el modelo que recibe los datos después de la tranformación para realizar predicciones. Finalmente, estos pipelines pueden resultar muy útiles a la hora de calibrar y comparar modelos, pues se tiene la certeza de que los datos de entrada son los mismos para todos. Incluso, pueden ser utilizados para realizar validación cruzada utilizando GridSerchCV o RandomizedSerchCV. Así mismo, pueden ser exportados para llevar los modelos a producción por medio de la serialización de estos en archivos .pkl o .joblib. 

La librería Scikit Learn cuenta con una API para la creación de pipelines en la que pueden ser utilizados diferentes pasos para la transformación de los datos que serán aplicados secuencialmente. Note que estos pasos implementan los métodos **fit** y **transform** para ser invocados desde el pipeline. Por otro lado, los modelos que serán la parte final del proceso de automatización solo cuentan con método fit. Una vez construido el modelo es posible serializar este haciendo uso de la función **dump** de la librería joblib, para posteriormente deserializar, cargar (mediante la función **load**) y utilizar el modelo en cualquier otra aplicación o ambiente. Tenga en cuenta que la serialización de un modelo solo incluye la estructura y configuraciones realizadas sobre el pipeline, más no las instancias de los objetos que lo componen. Pues estos son provistos por la librería, por medio de la importación, en cualquiera que sea su ambiente de ejecución. Esto significa que si usted construye transformaciones personalizadas, debe incluir por separado estas en el ambiente donde cargará y ejecutará el modelo una vez sea exportado, ya que estas no están incluidas en la serialización. 

Basándose en los pasos realizados para la calibración de su modelo de regresión del laboratorio 2. Construya un pipeline que incluya todos los pasos necesarios para transformar los datos desde el archivo fuente para que estos puedan ser utilizados para realizar predicciones.

A continuación puede encontrar algunos artículos que pueden ser de utilidad para la construcción de pipelines. 
<br>
<br>
[Scikit-learn Pipeline Tutorial with Parameter Tuning and Cross-Validation](https://towardsdatascience.com/scikit-learn-pipeline-tutorial-with-parameter-tuning-and-cross-validation-e5b8280c01fb)
<br>
[Data Science Quick Tip #003: Using Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-003-using-scikit-learn-pipelines-66f652f26954)
<br>
[Data Science Quick Tip #004: Using Custom Transformers in Scikit-Learn Pipelines!](https://towardsdatascience.com/data-science-quick-tip-004-using-custom-transformers-in-scikit-learn-pipelines-89c28c72f22a)
<br>
[Creating custom scikit-learn Transformers](https://www.andrewvillazon.com/custom-scikit-learn-transformers/)

## Construcción de la API
Para esta sección se recomienda utilizar el IDE enfocado a desarrollo para realizar la construcción de la API. Tenga en cuenta que la siguiente guía está desarrollada para el framework recomendado ([FastApi](https://fastapi.tiangolo.com/) ), sin embargo, el uso de este no es obligatorio.  

1. Crear un proyecto nuevo proyecto de Python con el nombre Lab 2 - API 
2. Instalar las dependencias necesarias para la construcción de la API. 
    - Instalar el framework ingresando los siguientes comandos en la terminal. Más información en la [documentación de instalación](https://fastapi.tiangolo.com/#installation). La dependeincia uvicorn corresponde con un servidor ASGI que simula el ambiente de producción. 
    ``` 
    pip install fastapi
    pip install "uvicorn[standard]" 
    ```
    
3. En un archivo main.py crear la API básica mostrada en la [documentación](https://fastapi.tiangolo.com/#installation) 
     ``` 
    from typing import Optional
    
    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/")
    def read_root():
        return {"Hello": "World"}
    
    
    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None):
        return {"item_id": item_id, "q": q}
    ``` 
    
4. Correr el servidor ingresando el siguiente comando en la terminal y verificar que el funcionamiento es correcto
     ``` 
    uvicorn main:app --reload
     ``` 
     
5. Puede verificar que la API está creada correctamente mediante la consulta de la documentación
     ```
    http://127.0.0.1:8000/docs
     ```
     
6. Crear en una archivo DataModel.py la clase que simboliza un registro de los datos que se recibirá por parte del cliente. En este caso, estos coinciden con todos los datos originales en los archivos "csv" compartidos, sin incluir la variable a predecir (variable objetivo). 
La librería [pydantic](https://pydantic-docs.helpmanual.io/) realiza sugerencias de tipo, en tiempo de ejecución y provee mensajes de error amigables con el usuario cuando los datos son inválidos.

    ```
    from pydantic import BaseModel

    class DataModel(BaseModel):
    
    # Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
        year: float
        km_driven: float
        .. COMPLETAR CON LAS VARIABLES EXPLICATIVAS DE LOS DATOS DEL LABORATORIO 2
    
    #Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
        def columns(self):
            return ["year","km_driven", ...]

    ```

7. Crear un archivo PredictionModel.py que contiene la clase Model la cual representa el modelo que será cargado. En el constructor se crea una instacia del modelo con base en el archivo joblib. 
Adicionalmente, se crea una función para realizar las predicciones.
    ```
    from joblib import load

    class Model:
    
        def __init__(self,columns):
            self.model = load("assets/modelo.joblib")
    
        def make_predictions(self, data):
            result = self.model.predict(data)
            return result
    ```
    
8. En el archivo main.py se debe crear una función encargada de tomar los datos recibidos en el cuerpo de la petición y transformarlos en un dataframe para que estos puedan ser recibidos por el modelo. Posteriormente, se crea una instancia del modelo y se realizan las predicciones. Esta instancia del modelo se puede crear utilizando la clase Model creada en el paso anterior o directamente como se muestra en el siguiente código. 
    ```
    @app.post("/predict")
    def make_predictions(dataModel: DataModel):
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
        df.columns = dataModel.columns()
        model = load("assets/modelo.joblib")
        result = model.predict(df)
        return result
    ```

9. Despliegue la API de nuevo y verifique el funcionamiento. Realice las correcciones que sean necesarias para garantizar el buen funcionamiento. 


## Lo que debió construir!!

1. Repositorio público de GitHub incluyendo lo siguiente:  
1.1. Script de Jupyter Lab con la construcción del pipeline y la explicación de cada uno de los pasos que lo componen.  
1.2. Código fuente de la API con los endpoints descritos en el enunciado.   
1.3. Archivo README con las instrucciones de despliegue de la API en un ambiente local y ejemplos de consumo de de cada uno de los endpoints.   
2. Escenarios de prueba:  
2.1. Al menos 2 escenarios de prueba de la API variando los datos de entrada. Debe incluir escenarios de prueba exitosos que generen predicciones coherentes así como escenarios de prueba erróneos en los que se produzcan predicciones incoherentes o errores de ejecución. Para cada uno de estos escenarios debe especificar los datos usados en formato JSON . Mencionar brevemente por qué el resultado obtenido se considera coherente y en caso contrario analizar por qué se produce el error.  
2.2. Párrafo corto en donde se exponga una estrategia a desarrollar para mitigar incoherencias en la predicción y errores de ejecución. 


## Puntos deseables

1. Construir e integrar en el pipeline transformadores personalizados. Garantizar que el proceso completo sea el deseado.
2. Desplegar la API en un servidor gratuito para que pueda ser consumido desde internet mediante su URL. 
3. Implementar la estrategia de mitigación de errores identificados en los escenarios de prueba y que fueron documentados en el entregable correspondiente.

## Evaluación de resultados

La segunda etapa del proyecto incluye estos elementos y es ahí donde serán evaluadas el desarrollo de estas actividades!!

