# KeepCoding-IA-I-despliegue-algoritmos-Practice

Hay que tener en cuenta que el token utilizado para generar todo NO funciona, con lo cual es imposible reproducir todo sin un token valido y vigente

La práctica está en:
[masterclass_despliegue_practica_jca](./masterclass_despliegue_practica_jca.ipynb)

Se crearon las siguiente archivos con código:
- [common](./common.py) Lee y guarda archivos PKL, y también muestra imágenes

- [text_preprocess](./text_preprocess.py) Contiene todas las funciones para realizar el preprocesado del texto
- [bag_of_words](./bag_of_words.py) Extrae features usando BoW


- [mlflow_params](./mlflow_params.py) Corre desde consola los modelos a entrenar que se pasan como parámetro

- [fastapi_mlmodels](./fastapi_mlmodels.py) Sirve para obtener resultados del entrenamiento que se utilizarán en llamadas de FastAPI
- [fastapi_hf](./fastapi_hf.py) Contiene todos los endpoints utilizados en FastAPI, incluidos los de ML y también HF



