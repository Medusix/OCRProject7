# %% imports libs
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% settings
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

# %% import raw data
path = "Dataset"
data = pd.read_csv(os.path.join(path, 'application_train.csv'))
#data_test = pd.read_csv(os.path.join(path, 'application_test.csv'))
print(data.shape)

print(data['TARGET'])
# %% Séparation du dataset en train et test sets
# Ne disposant pas de valeur target dans le datasetapplication_test, nous allons le mettre de côté et nous servir uniquemement du train set que nous allons séparer en train et en test set
y_train = data.pop('TARGET')
data_train = data

X_train, X_test, y_train, y_test = train_test_split(data_train, y_train, test_size = .25, random_state=10)

# %% encoding of some categorial features

le = LabelEncoder()

for col in X_train:
    if X_train[col].dtype == 'object':
        if len(list(X_train[col].unique())) <= 2:
            le.fit(X_train[col])
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join='inner', axis=1)
# %%
features = X_train.columns.tolist()
print(features)
# %% nettoyage de DAYS_EMPLOYED
X_train['DAYS_EMPLOYED_ANOM'] = X_train["DAYS_EMPLOYED"] == 365243
X_test['DAYS_EMPLOYED_ANOM'] = X_test["DAYS_EMPLOYED"] == 365243
X_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
X_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

# nettoyage de DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH
X_train['DAYS_BIRTH'] = - X_train['DAYS_BIRTH']
X_train['DAYS_EMPLOYED'] = - X_train['DAYS_EMPLOYED']
X_train['DAYS_REGISTRATION'] = - X_train['DAYS_REGISTRATION']
X_train['DAYS_ID_PUBLISH'] = - X_train['DAYS_ID_PUBLISH']
X_test['DAYS_BIRTH'] = - X_test['DAYS_BIRTH']
X_test['DAYS_EMPLOYED'] = - X_test['DAYS_EMPLOYED']
X_test['DAYS_REGISTRATION'] = - X_test['DAYS_REGISTRATION']
X_test['DAYS_ID_PUBLISH'] = - X_test['DAYS_ID_PUBLISH']

# feature selection - Voir si on réalise une feature sélection pour améliorer la qualité de prédiciotn du modèle
'''list_features_categorielles = C_train.select_dtypes('object').columns.tolist()
df_ex_target_ex_cat = C_train[[col for col in C_train.columns if col not in list_features_categorielles]]
df_ex_target_ex_cat = df_ex_target_ex_cat.dropna(axis=0)
df_target = df_ex_target_ex_cat['TARGET']
df_ex_target_ex_cat = df_ex_target_ex_cat.drop(columns=['TARGET'])

kbest = SelectKBest(score_func=f_classif, k=20)
kbest.fit(df_ex_target_ex_cat, df_target)
selected_features = list(df_ex_target_ex_cat.columns[kbest.get_support()])'''
# %% export des dataset nettoyés
X_train.to_csv(os.path.join('Dataset', 'Data clean', 'X_train.csv'), index=False)
y_train.to_csv(os.path.join('Dataset', 'Data clean' , 'y_train.csv'), index=False)
X_test.to_csv(os.path.join('Dataset', 'Data clean' , 'X_test.csv'), index=False)
y_test.to_csv(os.path.join('Dataset', 'Data clean' , 'y_test.csv'), index=False)
# %%
