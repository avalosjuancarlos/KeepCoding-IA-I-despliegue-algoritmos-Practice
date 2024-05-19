#%%writefile fastapi_hf.py
from typing import Annotated
from fastapi import FastAPI, Query
import fastapi_mlmodels as ml
from transformers import pipeline
 
# creamos los pipeline de HF
sentiment_pipeline = pipeline("sentiment-analysis")
classifier = pipeline("summarization", 
                      model='josmunpen/mt5-small-spanish-summarization')

app = FastAPI()

@app.get('/')
def root():
  return {'message': 'Estas en la práctica de MLOps!!!!!',
          'status': '200 - OK'}
 
@app.get('/saluda')
def hello():
   return {'Message': 'Hola soy Juan'}

@app.get('/entrenamiento/mejor_modelo')
def best_model():
  return ml.get_best_model()

@app.get('/entrenamiento/resultados/')
def read_items(q: Annotated[list[str] | None, Query()] = None):
  results = {}
  print(f'models: {q}')
  if q != None:
    print("pasa por results con modelos")
    results = ml.get_results(q)
  else:
    print("usa el default")
    results = ml.get_results()
    print(results)
  
  return { "results": f'{results}' }

@app.get('/hf/english_sentiment/')
def sentiment(message):
  data = [message]
  return sentiment_pipeline(data)
  
@app.get('/hf/resumen/')
def summary(message):
  return classifier(message)
