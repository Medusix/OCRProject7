# %% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import mlflow
from mlflow.models import infer_signature
from mlflow_tools import mlflow_eval_metrics, mlflow_log_experiment, mlflow_log_model

# from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import shap


# %% Settings
shap.initjs()
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

# Sélection des modèle et datasets
METHOD = "Logistic Regression"  # "Logistic Regression" or "Random Forest" or "Gradient Boosting"
DATASET = "undersampled"  # "original "ou "undersampled" ou "oversampled"
RUN_NAME = METHOD + " avec dataset " + DATASET
FEATURE_IMPORTANCE = True
LOG_MLFLOW = True


# %% Functions
def addition(a=0, b=0):
    '''Retourne la somme de a et b.
    Fonction uniquement créée pour tester la bonne mise en place de Pytest lors d'un push github.

    Parameters:
        - a : int : Première valeur à additionner.
        - b : int : Deuxième valeur à additionner.

    Returns:
        - somme : int : la somme de a et b.
    '''

    return a+b


# %% imports data
# Train set
if DATASET == "original":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train.parquet'))
elif DATASET == "oversampled":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_oversampled.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_oversampled.parquet'))
elif DATASET == "undersampled":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_undersampled.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_undersampled.parquet'))

X_train.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)

# Test set
X_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_test.parquet'))
y_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_test.parquet'))

X_test.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

# %% Suppression des valeurs manquantes dans le cadre de la régression logistique
# Valeurs manquantes train
'''train = pd.concat([X_train, y_train], axis=1)

print(train.shape)
train.dropna(inplace=True)
print(train.shape)
y_train = train.pop('TARGET')
X_train = train
# Valeurs manquantes test
test = pd.concat([X_test, y_test], axis=1)
test.dropna(inplace=True)
y_test = test.pop('TARGET')
X_test = test
'''
# %%
all_feat = list(X_train.columns.values)
describe = X_test.describe().T
describe = describe.sort_values(by="max", axis=0)
print(describe)
describe = describe.sort_values(by="min", axis=0)
print(describe)


# %% Modélisations
if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='credit_score_classification')

    if METHOD == "Logistic Regression":

        # Modélisation Logistic Regression
        # run_name = "Run Logistic regression"

        with mlflow.start_run(run_name=RUN_NAME):
            # HalvingRandomSearchCV
            log_reg = LogisticRegression()

            params_lr = {"C": np.logspace(0.1, 2, 4)}

            halving_grid_lr = HalvingRandomSearchCV(estimator=log_reg,
                                                    param_distributions=params_lr,
                                                    cv=5,
                                                    scoring='accuracy',
                                                    error_score=0,
                                                    n_jobs=-1
                                                    )
            halving_grid_lr.fit(X_train, y_train)

            # Modélisation, prédictions et performances
            model = LogisticRegression(**halving_grid_lr.best_params_)
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)
            # accuracy, auc, f1 = mlflow_tools.mlflow_eval_metrics(y_test, predictions)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # mlflow_tools.mlflow_log(model, run_name, halving_grid_lr.best_params_, accuracy, auc, f1)
                # log a mlflow experiment
                mlflow_log_experiment(model, RUN_NAME, halving_grid_lr.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test, y_test, halving_grid_lr.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)

    elif METHOD == "Random Forest":
        # Modélisation Logistic Regression
        # run_name = "Run Random Forest Classifier"

        with mlflow.start_run(run_name=RUN_NAME):
            # HalvingRandomSearchCV
            rfc = RandomForestClassifier()

            params_rfc = {
                "n_estimators": np.linspace(10, 150, 5, dtype=int),
                "max_depth": np.linspace(5, 50, 4, dtype=int),
                "min_samples_split": np.linspace(5, 50, 4, dtype=int)
                }

            halving_rand_rfc = HalvingRandomSearchCV(
                estimator=rfc,
                param_distributions=params_rfc,
                cv=5,
                scoring='accuracy',
                error_score=0,
                n_jobs=-1
            )
            halving_rand_rfc.fit(X_train, y_train)

            # Modélisation, prédictions et performances
            model = RandomForestClassifier(**halving_rand_rfc.best_params_)
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # MLFlow log
                mlflow_log_experiment(model, RUN_NAME, halving_rand_rfc.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test, y_test, halving_rand_rfc.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)

    elif METHOD == "Gradient Boosting":
        # Modélisation Gradient Boosting Classifier
        # run_name = "Run Gradient Boosting Classifier"

        with mlflow.start_run(run_name=RUN_NAME):
            # Gradient Boosting
            gbc = GradientBoostingClassifier()

            params_gbc = {"learning_rate": np.logspace(-2, 0, 4),
                          "n_estimators": np.linspace(10, 150, 5, dtype=int),
                          "min_samples_split": np.linspace(5, 50, 4, dtype=int)
                          }

            halving_rand_gbc = HalvingRandomSearchCV(
                estimator=gbc,
                param_distributions=params_gbc,
                cv=5,
                scoring='accuracy',
                error_score=0,
                n_jobs=-1
                )
            halving_rand_gbc.fit(X_train, y_train)

            # Modélisation, prédictions et performances
            model = GradientBoostingClassifier(**halving_rand_gbc.best_params_)
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # MLFlow log
                mlflow_log_experiment(model, RUN_NAME, halving_rand_gbc.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test, y_test, halving_rand_gbc.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)
    # %% Feature importance globale
    if FEATURE_IMPORTANCE:
        features = X_train.columns
        if METHOD == "Logistic Regression":
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)

            shap.plots.beeswarm(shap_values, features=features)
        elif METHOD in ("Random Forest", "Gradient Boosting"):
            explainer = shap.TreeExplainer(model)
            explainer.expected_value = explainer.expected_value[0]
            shap_values = explainer.shap_values(X_train)

            newCmap = LinearSegmentedColormap.from_list("", ['#c4cfd4', '#3345ea'])
            cmap = plt.get_cmap('tab10')
            max_feat_display = 14
            plt.title(f"Contributions des {max_feat_display} principales features au score global", fontsize=14)
            # shap.summary_plot(shap_values, X_train, plot_type="violin", color=newCmap, max_display=10)
            # shap.summary_plot(shap_values, X_train, plot_type="violin", max_display=10)
            shap.summary_plot(
                shap_values,
                features=features,
                max_display=max_feat_display,
                plot_type='bar',  # 'violin' ou 'dot' ou 'bar'
                plot_size=(12, 6))
# %%
