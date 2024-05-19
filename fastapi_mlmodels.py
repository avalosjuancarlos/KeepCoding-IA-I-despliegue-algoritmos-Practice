#%%writefile fastapi_mlmodels.py
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

# Configuramos una semilla que va a ser utilizada para entrenar todos los modelos
global_seed = 42

# Creamos un array de modelos an entrenar
models_names = ['RF', 'LS', 'KN', 'DT', 'SV', 'GB']
models = []
models_dict = {'RF': "Random Forest",
              'LS': "LASSO",
              'KN': "KNN",
              'DT': "Decision Tree",
              'SV': "SVM",
              'GB':"Gradient Boosting"
              }

if "RF" in models_names:
  models.append((models_dict["RF"], RandomForestClassifier(random_state=global_seed)))
if "LS" in models_names:
  models.append((models_dict["LS"], lm.LogisticRegression(random_state=global_seed)))
if "KN" in models_names:
  models.append((models_dict["KN"], KNeighborsClassifier()))
if "DT" in models_names:
  models.append((models_dict["DT"], DecisionTreeClassifier(random_state=global_seed)))
if "SV" in models_names:
  models.append((models_dict["SV"], SVC(random_state=global_seed)))
if "GB" in models_names:
  models.append((models_dict["GB"], GradientBoostingClassifier(random_state=global_seed)
))
# Creamos arrays para almacenar los resultados
results = {}
names = []

# setemos en 0 los mejores resultados
best_mean_result = 0
best_std_result = 0

for name, model in models:
  kfold = StratifiedKFold()
  cv_results = cross_val_score(model, x_features_train, y_labels_train, scoring='accuracy', cv=kfold)
  names.append(name)
  print(name + ": mean(accuracy)=" + str(round(np.mean(cv_results), 3)) + ", std(accuracy)=" + str(round(np.std(cv_results), 3)))
  results[name] = {
    'cv_results': cv_results,
    'mean': np.mean(cv_results),
    'std': np.std(cv_results)
  }

  if (best_mean_result < np.mean(cv_results)) or \
    ((best_mean_result == np.mean(cv_results)) and (best_std_result > np.std(cv_results))):
    best_mean_result = np.mean(cv_results)
    best_std_result = np.std(cv_results)
    best_model = { 'model': model, 'name': name }


# end preload ----------------------------------

# Obtenemos el nombre del mejor modelo
def get_best_model():
  return best_model["name"]


# Obtenemos los resultados de todos los modelos o de los que elijamos
def get_results(models_names = ['RF', 'LS', 'KN', 'DT', 'SV', 'GB']):
  user_results = {}

  models_full_name = []
  for name in models_names:
      if name in models_dict:
          models_full_name.append(models_dict[name])

  for name in models_full_name:
    if name in results:
      user_results[name] = results[name]

  return user_results
