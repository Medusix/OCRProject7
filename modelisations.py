# %% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import shap

import mlflow
from mlflow.models import infer_signature
from mlflow_tools import mlflow_eval_metrics, mlflow_log_experiment, mlflow_log_model

# %% Settings
shap.initjs()
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

# %% Fonctions copiées/collées ici en attendant de résoudre le problème de dépendance
'''def mlflow_eval_metrics(y_true, y_pred, fn_to_fp=10):
    ''''''Calculates classification metrics.

    -------------
    Parameters:
    - y_true : 1d array-like, or label indicator array / sparse matrix : Ground truth (correct) labels.
    - y_pred : 1d array-like, or label indicator array / sparse matrix : Predicted labels, as returned by a classifier.
    - fn_to_fp : int : Pondération du coût d'un Faux négatif par rapport à un faux positif. Utilisé pour calculer le coût métier.

    -------------
    Returns:
    - accuracy : float : Fraction of correctly classified samples.
    - auc : float : Area Under the Curve.
    - f1 : float : F1 score of the positive class in binary classification.
    - cout_metier :
    ''''''
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpr)
    # Calcul d'un coût métier pondérant un Faux négatif à 10 fois un Faux positif.
    # Coût métier = (Nombre de faux négatifs x pondération + Nombre de faux positifs x 1) / nombre de prédictions 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cout_metier = (fn * fn_to_fp + fp * 1)/y_pred.shape[0]

    return accuracy, auc_, f1, cout_metier


def mlflow_log(model, run_name, params, accuracy, auc, f1, cout_metier):
    ''''''Logs a run via MLFlow.

    -------------
    Parameters:
    - model : sklearn model : Instance of a sklearn model among:
                                - sklearn.ensemble.RandomForestClassifier
                                - skleanr.ensemble.GradientBoostingClassifier
                                - sklearn.linear_model.LogisticRegression
    - run_name : str : Name of the MLFlow run.
    - params: dictionnaire : Dictionary of params as expressed by the cross_validation hyper parameter optimizer .best_params_.
                        - RandomForestClassifier: ('max_depth', 'min_samples_split', 'n_estimators')
                        - LogisticRegression: ('penalyt', 'C')
    - accuracy : float : Fraction of correctly classified samples.
    - auc : float : Area Under the Curve.
    - f1 : float : F1 score of the positive class in binary classification.
    ''''''
    if isinstance(model, RandomForestClassifier):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('max_depth', params['max_depth'])
        mlflow.log_param('min_samples_split', params['min_samples_split'])
        mlflow.log_param('n_estimators', params['n_estimators'])

        # mlflow.sklearn.save_model(model, "RandomForestClassifier1")

    elif isinstance(model, LogisticRegression):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('C', params['C'])

        # mlflow.sklearn.save_model(model, "LogisticRegression")

    elif isinstance(model, GradientBoostingClassifier):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('learning_rate', params['learning_rate'])
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_param('min_samples_split', params['min_samples_split'])

        # mlflow.sklearn.save_model(model, "GradientBoostingRegressor")

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('auc', auc)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('coût_métier', cout_metier)'''


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


# %% Sélection des modèle et datasets
METHOD = "Logistic Regression"
DATASET = "undersampled"  # "base "ou "undersampled" ou "oversampled"
RUN_NAME = METHOD + " avec dataset " + DATASET
print(RUN_NAME)

# %% imports data
# Train set
if DATASET == "base":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train.parquet'))
elif DATASET == "oversampled":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_oversampled.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_oversampled.parquet'))
elif DATASET == "undersampled":
    X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_undersampled.parquet'))
    y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_undersampled.parquet'))

# Test set
X_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_test.parquet'))
y_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_test.parquet'))

'''features_to_remove = [
    "INSTAL_PAYMENT_PERC_MAX",
    "INSTAL_PAYMENT_PERC_MEAN",
    "INSTAL_PAYMENT_PERC_SUM",
    "PREV_APP_CREDIT_PERC_MEAN",
    "PREV_APP_CREDIT_PERC_MAX",
    "REFUSED_APP_CREDIT_PERC_MEAN",
    "REFUSED_APP_CREDIT_PERC_MAX"
    ]
X_train = X_train.drop(columns=features_to_remove)
X_test = X_test.drop(columns=features_to_remove)'''

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
        # Suppression des valeurs manquantes dans le cadre de la régression logistique
        # Valeurs manquantes train
        train = pd.concat([X_train, y_train], axis=1)

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
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            # accuracy, auc, f1 = mlflow_tools.mlflow_eval_metrics(y_test, predictions)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

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
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            # MLFlow log
            mlflow_log_experiment(model, RUN_NAME, halving_rand_rfc.best_params_, accuracy, auc_, f1, cout_metier)
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
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            # MLFlow log
            mlflow_log_experiment(model, RUN_NAME, halving_rand_gbc.best_params_, accuracy, auc_, f1, cout_metier)
# %%
print(X_train.shape)
# print(X_train.info())
print("Nan values in X_train:", X_train.isna().sum().sum())
# %% Feature importance globale
if METHOD == "Logistic Regression":
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap.plots.beeswarm(shap_values)
elif METHOD == "Random Forest" or METHOD == "Gradient Boosting":
    explainer = shap.TreeExplainer(model)
    explainer.expected_value = explainer.expected_value[0]
    shap_values = explainer.shap_values(X_train)

    cmap = plt.get_cmap('tab10')
    plt.title("Contributions principales au score", fontsize=14)
    shap.summary_plot(shap_values, X_train, plot_type="violin", color=cmap.colors[0], max_display=10)
# %%
print(X_train.info())
# %%
