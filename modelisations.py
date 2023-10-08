# %% Imports
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score

import mlflow
#import mlflow_tools => en attendant de régler le problème.

# %% Fonctions copiées/collées ici en attendant de résoudre le problème de dépendance
def mlflow_eval_metrics(y_true, y_pred):
    '''Calculates classification metrics.
    
    -------------
    Parameters:
    - y_true : 1d array-like, or label indicator array / sparse matrix : Ground truth (correct) labels.
    - y_pred : 1d array-like, or label indicator array / sparse matrix : Predicted labels, as returned by a classifier.
    
    -------------
    Returns:
    - accuracy : float : Fraction of correctly classified samples.
    - auc : float : Area Under the Curve.
    - f1 : float : F1 score of the positive class in binary classification.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    #auc_ = auc(y_true, y_pred)
    auc_ = roc_auc_score(y_true, y_pred)
    roc_auc_score
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, auc_, f1

def mlflow_log(model, run_name, params, accuracy, auc, f1):
    '''Logs a run via MLFlow.

    -------------
    Parameters:
    - model : sklearn model : Instance of a sklearn model among:
                                - sklearn.ensemble.RandomForestClassifier
                                - sklearn.linear_model.LogisticRegression
    - run_name : str : Name of the MLFlow run.
    - params: tuple : Tuple of params as expressed by the cross_validation hyper parameter optimizer .best_params_.
                        - RandomForestClassifier: ('max_depth', 'min_samples_split', 'n_estimators')
                        - LogisticRegression: ('penalyt', 'C')
    - accuracy : float : Fraction of correctly classified samples.
    - auc : float : Area Under the Curve.
    - f1 : float : F1 score of the positive class in binary classification.
    '''
    if isinstance(model, sklearn.ensemble.RandomForestClassifier):
        print(run_name)
        print(f"\tAccuracy: {accuracy:.03f}")
        print(f"\tAUC: {auc:.03f}")
        print(f"\tF1_score: {f1:.03f}")
        
        mlflow.log_param('max_depth', params['max_depth'])
        mlflow.log_param('min_samples_split', params['min_samples_split'])
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('auc', auc)
        mlflow.log_metric('f1', f1)
    
    elif isinstance(model, sklearn.linear_model.LogisticRegression):
        print(run_name)
        print(f"\tAccuracy: {accuracy:.03f}")
        print(f"\tAUC: {auc:.03f}")
        print(f"\tF1_score: {f1:.03f}")
        
        mlflow.log_param('C', params['C'])
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('auc', auc)
        mlflow.log_metric('f1', f1)
# %% settings
def set_settings():
    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500

# Créé pour faire du 
# test unitaire
set_settings()

def addition(a=0, b=0):
    '''Retourne la somme de a et b.
    Fonction uniquement créée pour tester la bonne mise en place de Pytest lors d'un push github.
    
    Parameters:
        - a : int : Première valeur à additionner.
        - b : int : Deuxième valeur à additionner.
    
    Returns:
        - somme : int : la somme de a et b.
    '''

    return(a+b)

# %% imports data
X_train = pd.read_csv(os.path.join('Dataset', 'Data clean', 'X_train.csv'))
X_test = pd.read_csv(os.path.join('Dataset', 'Data clean', 'X_test.csv'))
y_train = pd.read_csv(os.path.join('Dataset', 'Data clean', 'y_train.csv'))
y_test = pd.read_csv(os.path.join('Dataset', 'Data clean', 'y_test.csv'))

# %% Modélisations

if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='credit_score_classification')
    
    method = "randomforest" #à revoir. Semble s'appeler via le terminal avec des paramètres => method: str = sys.argv[1] if len(sys.argv) > 1 else 'manuel'
    
    if method == "logreg":
        # Suppression des valeurs manquantes dans le cadre de la logistic_regression
        # Valeurs manquantes train
        train = pd.concat([X_train, y_train], axis=1)
        train.dropna(inplace=True)
        y_train = train.pop('TARGET')
        X_train = train
        # Valeurs manquantes test
        test = pd.concat([X_test, y_test], axis=1)
        test.dropna(inplace=True)
        y_test = test.pop('TARGET')
        X_test = test
        
        # Modélisation Logistic Regression
        run_name="Run Logistic regression"
        
        with mlflow.start_run(run_name=run_name):
            # HalvingRandomSearchCV
            log_reg = LogisticRegression()
            
            params_lr = {"C": np.logspace(0.1, 2, 4)}
            
            halving_grid_lr = HalvingRandomSearchCV(estimator=log_reg, 
                                            #param_grid=params_lr,
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
            #accuracy, auc, f1 = mlflow_tools.mlflow_eval_metrics(y_test, predictions)
            accuracy, auc, f1 = mlflow_eval_metrics(y_test, predictions)

            #mlflow_tools.mlflow_log(model, run_name, halving_grid_lr.best_params_, accuracy, auc, f1)
            mlflow_log(model, run_name, halving_grid_lr.best_params_, accuracy, auc, f1)
    if method == "randomforest":
        # Modélisation Logistic Regression
        run_name="Run Random Forest Classifier"
        
        with mlflow.start_run(run_name=run_name):
            #HalvingRandomSearchCV
            rfc = RandomForestClassifier()
            
            params_rfc = {"n_estimators" : np.linspace(10,150,5,dtype = int),
                          "max_depth" : np.linspace(5,50,4,dtype = int),
                          "min_samples_split" : np.linspace(5,50,4,dtype = int)
                         }
            
            halving_rand_rfc = HalvingRandomSearchCV(estimator=rfc, 
                                            #param_grid=params_lr,
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
            accuracy, auc, f1 = mlflow_eval_metrics(y_test, predictions)
            
            #MLFlow log
            mlflow_log(model, run_name, halving_rand_rfc.best_params_, accuracy, auc, f1)
            
# %%
