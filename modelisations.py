# %% Imports
import os
from pickle import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

import shap

import mlflow
from mlflow.models import infer_signature
from mlflow_tools import mlflow_eval_metrics, mlflow_log_experiment, mlflow_log_model

import Data_prep
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# %% Settings
shap.initjs()
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

# mlflow.set_tracking_uri('http://ec2-13-49-44-23.eu-north-1.compute.amazonaws.com:5000')

METHOD = "Logistic Regression"  # "Logistic Regression" or "Random Forest" or "Gradient Boosting"
DATASET = "undersampled"  # "original "ou "undersampled" ou "SMOTE"
RUN_NAME = METHOD + "_" + DATASET
FEATURE_IMPORTANCE = False
LOG_MLFLOW = True
RUN_EVIDENTLY = False


# %% Functions
def cout_metier(y_true, y_pred, fn_to_fp=10):
    '''Fonction de calcul de coût métier, utilisé pour optimiser la recherche RandomSearch.
        Optimisation en surpondérant par un facteur fn_to_fp le poids d'un Faux Négatif par rapport à un Faux positif.
    '''
    _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    cout = (fn * fn_to_fp + fp * 1)/y_pred.shape[0]
    return cout


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


# %% Modélisations
if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='credit_score_classification')

    # Data preparartion - Train set
    if DATASET == "original":
        X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train.parquet'))
        y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train.parquet'))
    elif DATASET == "oversampled":
        X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_oversampled.parquet'))
        y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_oversampled.parquet'))
    elif DATASET == "undersampled":
        X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_undersampled.parquet'))
        y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_undersampled.parquet'))
    elif DATASET == "SMOTE":
        X_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_train_SMOTE.parquet'))
        y_train = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_train_SMOTE.parquet'))

    X_train.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)
    y_train.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)

    # Data preparartion - Test set
    X_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join('Dataset', 'Data clean', 'y_test.parquet'))

    X_test.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)
    y_test.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)

    # Data scaling and scaler parameters save
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    dump(scaler, open('scaler.pkl', 'wb'))

    # Scorer custom
    custom_scorer = make_scorer(cout_metier, greater_is_better=False)

    if METHOD == "Logistic Regression":
        with mlflow.start_run(run_name=RUN_NAME):
            # HalvingRandomSearchCV
            log_reg = LogisticRegression()

            params_lr = {
                "C": np.logspace(0.1, 2, 4),
                "solver": ['newton-cg', 'newton-cholesky', 'sag']}

            halving_grid_lr = HalvingRandomSearchCV(estimator=log_reg,
                                                    param_distributions=params_lr,
                                                    cv=5,
                                                    scoring=custom_scorer,
                                                    error_score=0,
                                                    n_jobs=-1
                                                    )
            halving_grid_lr.fit(X_train_scaled, y_train)

            # Modélisation, prédictions et performances
            model = LogisticRegression(**halving_grid_lr.best_params_)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            # accuracy, auc, f1 = mlflow_tools.mlflow_eval_metrics(y_test, predictions)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # mlflow_tools.mlflow_log(model, run_name, halving_grid_lr.best_params_, accuracy, auc, f1)
                # log a mlflow experiment
                mlflow_log_experiment(model, RUN_NAME, halving_grid_lr.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test_scaled, y_test, halving_grid_lr.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)

    elif METHOD == "Random Forest":
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
                scoring=custom_scorer,
                error_score=0,
                n_jobs=-1
            )
            halving_rand_rfc.fit(X_train_scaled, y_train)

            # Modélisation, prédictions et performances
            model = RandomForestClassifier(**halving_rand_rfc.best_params_)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # MLFlow log
                mlflow_log_experiment(model, RUN_NAME, halving_rand_rfc.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test_scaled, y_test, halving_rand_rfc.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)

    elif METHOD == "Gradient Boosting":
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
                scoring=custom_scorer,
                error_score=0,
                n_jobs=-1
                )
            halving_rand_gbc.fit(X_train_scaled, y_train)

            # Modélisation, prédictions et performances
            model = GradientBoostingClassifier(**halving_rand_gbc.best_params_)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            accuracy, auc_, f1, cout_metier = mlflow_eval_metrics(y_test, predictions)

            if LOG_MLFLOW:
                # MLFlow log
                mlflow_log_experiment(model, RUN_NAME, halving_rand_gbc.best_params_, accuracy, auc_, f1, cout_metier)

                # artifacts logging
                mlflow.log_artifacts('Artifacts')

                # log a mlflow model
                mlsignature = infer_signature(X_test_scaled, y_test, halving_rand_gbc.best_params_)
                mlflow_log_model(model=model, signature=mlsignature)

    # Feature importance globale
    if FEATURE_IMPORTANCE:
        features = X_train.columns
        if METHOD == "Logistic Regression":
            # explainer = shap.Explainer(model, X_test)
            # shap_values = explainer(X_test)
            explainer = shap.Explainer(model, X_train_scaled, feature_names=features)
            shap_values = explainer(X_train_scaled)

            # Global feature importance
            plt.figure()
            plt.title('Global feature importance')
            shap.plots.beeswarm(shap_values)

            # Local feature importance
            for user in range(0, 10):
                plt.figure()
                plt.title(f'Local feature importance - Prediction {user}')
                shap.plots.bar(shap_values=shap_values[user], max_display=10)

        elif METHOD in ("Random Forest", "Gradient Boosting"):
            explainer = shap.TreeExplainer(model)
            explainer.expected_value = explainer.expected_value[0]
            shap_values = explainer.shap_values(X_test_scaled)

            newCmap = LinearSegmentedColormap.from_list("", ['#c4cfd4', '#3345ea'])
            cmap = plt.get_cmap('tab10')
            MAX_FEAT_DISPLAY = 14
            plt.title(f"Contributions des {MAX_FEAT_DISPLAY} principales features au score global", fontsize=14)
            # shap.summary_plot(shap_values, X_train_scaled, plot_type="violin", color=newCmap, max_display=10)
            # shap.summary_plot(shap_values, X_train_scaled, plot_type="violin", max_display=10)
            shap.summary_plot(
                shap_values,
                features=features,
                max_display=MAX_FEAT_DISPLAY,
                plot_type='bar',  # 'violin' ou 'dot' ou 'bar'
                plot_size=(12, 6))

    # Data Drift analysis
    if RUN_EVIDENTLY:
        x1, x2, _, _ = Data_prep.data_preparation(main_dataset_only=True, debug=False, new_data=True)
        new_data = pd.concat([x1, x2], axis=0)
        new_data.set_index(keys=['SK_ID_CURR'], drop=True, inplace=True)

        common_cols = list(set(new_data.columns).intersection(X_train.columns))
        new_data = new_data[common_cols]
        X_train = X_train[common_cols]

        data_drift_report = Report(metrics=[DataDriftPreset(),])
        data_drift_report.run(current_data=new_data, reference_data=X_train, column_mapping=None)

        data_drift_report.save_html("data_drift2.html")

# %% Analyse SHAP
