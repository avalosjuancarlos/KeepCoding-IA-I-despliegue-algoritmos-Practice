#%%writefile mlflow_params.py
import common
import sklearn.preprocessing as pr

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

import subprocess
import time

import mlflow

import argparse

# Corremos todos los modelos utilizando StratifiedKFlod y cross_val_score
def best_model(models_names, experiment_text, x_features_train, x_features_test, y_labels_train, y_labels_test):
  # Configuramos una semilla que va a ser utilizada para entrenar todos los modelos
  global_seed = 42

  # Creamos un array de modelos an entrenar
  # Como va a ser una revisión general de modelos, no se pone mucho enfasis
  # en configurar los diferentes parámetres
  models = []
  if "RF" in models_names:
    models.append(("Random Forest", RandomForestClassifier(random_state=global_seed)))
  if "LS" in models_names:
    models.append(("LASSO", lm.LogisticRegression(random_state=global_seed)))
  if "KN" in models_names:
    models.append(("KNN", KNeighborsClassifier()))
  if "DT" in models_names:
    models.append(("Decision Tree", DecisionTreeClassifier(random_state=global_seed)))
  if "SV" in models_names:
    models.append(("SVM", SVC(random_state=global_seed)))
  if "GB" in models_names:
    models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=global_seed)
  ))
  # Creamos arrays para almacenar los resultados
  results = []
  names = []

  # setemos en 0 los mejores resultados
  best_mean_result = 0
  best_std_result = 0

  mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000'])

  time.sleep(5)

  mlflow.set_experiment(experiment_text)

  for name, model in models:
    with mlflow.start_run() as run:
      kfold = StratifiedKFold()
      cv_results = cross_val_score(model, x_features_train, y_labels_train, scoring='accuracy', cv=kfold)
      results.append(cv_results)

      mlflow.log_metric('m1', np.mean(cv_results))
      mlflow.log_param('model', name)
      mlflow.log_param('accuracy', cv_results)
      mlflow.sklearn.log_model(model, 'clf_model')

      names.append(name)
      print(name + ": mean(accuracy)=" + str(round(np.mean(cv_results), 3)) + ", std(accuracy)=" + str(round(np.std(cv_results), 3)))

      if (best_mean_result < np.mean(cv_results)) or \
        ((best_mean_result == np.mean(cv_results)) and (best_std_result > np.std(cv_results))):
        best_mean_result = np.mean(cv_results)
        best_std_result = np.std(cv_results)
        best_model_name = name
        best_model = model

  return best_model, best_model_name



def main():
  parser = argparse.ArgumentParser(description='Argumentos de entrada en el main')
  parser.add_argument('--experiment_text', type=str, help='Nombre del entrenamiento (Experiment)')
  parser.add_argument('--models', nargs='+', type=str, help='Lista de nombres de models. [RF, LS, KN, DT, SV, GB].')

  args = parser.parse_args()
  experiment_text = args.experiment_text
  models_names = args.models

  best_model(models_names, experiment_text, 
  x_features_train, x_features_test, y_labels_train, y_labels_test)


# begin pre-load -------------------------------------------------------

base_path = ''
# obtenes la data guardada
preprocessed_data = common.load_pkl(base_path + 'preprocessed_data.pkl')
preprocessed_data.keys()

words_train = preprocessed_data['words_train']
words_test = preprocessed_data['words_test']
labels_train = preprocessed_data['labels_train']
labels_test = preprocessed_data['labels_test']

# obtenes la data guardada
bow_features = common.load_pkl(base_path + 'bow_features.pkl')
bow_features.keys()

features_train = bow_features['features_train']
features_test = bow_features['features_test']
vacabulary = bow_features['vocabulary']

# Normalización de Features
features_train = pr.normalize(features_train, axis=1)
features_test = pr.normalize(features_test, axis=1)

# creamos variables standard
x_features_train = features_train
y_labels_train = labels_train

x_features_test = features_test
y_labels_test = labels_test

# end preload ----------------------------------

# ejecutamos solo cuando el archivo es el principal
if __name__ == '__main__':
  main()
