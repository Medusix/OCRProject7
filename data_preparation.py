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

# %% Gestion du déséquilibre des classes 
if isinstance(y_train, pd.core.series.Series):
    y_train = y_train.to_frame()

nb_pos = y_train[y_train['TARGET'] == 1].shape[0]
nb_neg = y_train[y_train['TARGET'] == 0].shape[0]
print(f'Proportion de targets négatives: {round(100*nb_neg/(nb_pos+nb_neg),2)}%')


# Méthodes pour gérer l'imbalance:
# 1 - Collecter plus de données => Impossible dans le cadre de ce projet
# 2 - Utilisation d'une métrique adapté (AUROC) => met en évidence que le modèle prédit la classe principale
# 3 - Utilisation d'algorithmes différents => Les 3 algorithmes testés (Logistic regression, Random forest et Gradient Boosting) yield les même résultats
# Nous pourrions utiliser un modèle qui pénalise la mauvaise classification de la classe sous-représentée
# 4 - Resampling: Over-sampling
# 5 - Resampling: Under-sampling
# 6 - Créer des échantillons synthétiques

# Nous allons utiliser des méthodes de resampling pour améliorer la qualité de prédiction du modèle.

# Under-sampling: Nous allons réduire le nombre de target négatives afin d'avoir un équilibre entre les 2 classes
train = pd.concat([X_train, y_train], axis = 1)
train_pos = train.query("TARGET == 1")
train_neg = train.query("TARGET == 0")
train_neg = train_neg.sample(train_pos.shape[0])

train = pd.concat([train_pos, train_neg], axis=0)

nb_pos = train[train['TARGET'] == 1].shape[0]
nb_neg = train[train['TARGET'] == 0].shape[0]
print(f'Proportion de targets négatives (après under-sampling): {round(100*nb_neg/(nb_pos+nb_neg),2)}%')

y_train_undersampled = train.pop('TARGET')
X_train_undersampled = train

# Over-sampling: Nous allons multiplier le nombre d'individus dont la target est positive
train = pd.concat([X_train, y_train], axis = 1)
train_pos = train.query("TARGET == 1")
train_neg = train.query("TARGET == 0")

list_df = [train_pos for i in range(int(train_neg.shape[0]/train_pos.shape[0]))]
list_df.append(train_neg)
train = pd.concat(list_df, axis=0)

nb_pos = train[train['TARGET'] == 1].shape[0]
nb_neg = train[train['TARGET'] == 0].shape[0]
print(f'Proportion de targets négatives (après over-sampling): {round(100*nb_neg/(nb_pos+nb_neg),2)}%')

y_train_oversampled = train.pop('TARGET')
X_train_oversampled = train


# %% export des dataset nettoyés
# datasets originaux
'''X_train.to_csv(os.path.join('Dataset', 'Data clean', 'X_train.csv'), index=False)
y_train.to_csv(os.path.join('Dataset', 'Data clean', 'y_train.csv'), index=False)
X_test.to_csv(os.path.join('Dataset', 'Data clean', 'X_test.csv'), index=False)
y_test.to_csv(os.path.join('Dataset', 'Data clean', 'y_test.csv'), index=False)'''
X_train.to_parquet(os.path.join('Dataset', 'Data clean', 'X_train.parquet'), index=False)
y_train.to_parquet(os.path.join('Dataset', 'Data clean', 'y_train.parquet'), index=False)
X_test.to_parquet(os.path.join('Dataset', 'Data clean', 'X_test.parquet'), index=False)
y_test=y_test.to_frame()
y_test.to_parquet(os.path.join('Dataset', 'Data clean', 'y_test.parquet'), index=False)

# datasets under_sampled
'''X_train_undersampled.to_csv(os.path.join('Dataset', 'Data clean', 'X_train_undersampled.csv'), index=False)
y_train_undersampled.to_csv(os.path.join('Dataset', 'Data clean', 'y_train_undersampled.csv'), index=False)'''
X_train_undersampled.to_parquet(os.path.join('Dataset', 'Data clean', 'X_train_undersampled.parquet'), index=False)
y_train_undersampled = y_train_undersampled.to_frame()
y_train_undersampled.to_parquet(os.path.join('Dataset', 'Data clean', 'y_train_undersampled.parquet'), index=False)

# datasets over_sampled
'''X_train_oversampled.to_csv(os.path.join('Dataset', 'Data clean', 'X_train_oversampled.csv'), index=False)
y_train_oversampled.to_csv(os.path.join('Dataset', 'Data clean', 'y_train_oversampled.csv'), index=False)'''
X_train_oversampled.to_parquet(os.path.join('Dataset', 'Data clean', 'X_train_oversampled.parquet'), index=False)
y_train_oversampled = y_train_oversampled.to_frame()
y_train_oversampled.to_parquet(os.path.join('Dataset', 'Data clean', 'y_train_oversampled.parquet'), index=False)


# %%
