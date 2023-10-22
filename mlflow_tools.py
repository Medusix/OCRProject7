# %% Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve, confusion_matrix

import mlflow
import mlflow.sklearn


# %% Fonctions copiées/collées ici en attendant de résoudre le problème de dépendance
def mlflow_eval_metrics(y_true, y_pred, fn_to_fp=10):
    '''Calculates classification metrics.

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
    - cout_metier : float : Coût erreur calculé comme la somme pondéré du nombre de Faux négatifs (pondéré par fn_to_fp) et Faux positifs
                            rapporté au nombre de sample.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpr)
    # Calcul d'un coût métier pondérant un Faux négatif à 10 fois un Faux positif.
    # Coût métier = (Nombre de faux négatifs x pondération + Nombre de faux positifs x 1) / nombre de prédictions
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    cout_metier = (fn * fn_to_fp + fp * 1)/y_pred.shape[0]

    return accuracy, auc_, f1, cout_metier


def mlflow_log_experiment(model, run_name, params, accuracy, auc_, f1, cout_metier):
    '''Logs a run via MLFlow.

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
    '''
    if isinstance(model, RandomForestClassifier):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc_,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('max_depth', params['max_depth'])
        mlflow.log_param('min_samples_split', params['min_samples_split'])
        mlflow.log_param('n_estimators', params['n_estimators'])

        # mlflow.sklearn.save_model(model, "RandomForestClassifier1")

    elif isinstance(model, LogisticRegression):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc_,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('C', params['C'])

        # mlflow.sklearn.save_model(model, "LogisticRegression")

    elif isinstance(model, GradientBoostingClassifier):
        print(run_name)
        print(f"\tAccuracy: {round(accuracy,2)}")
        print(f"\tAUC: {round(auc_,2)}")
        print(f"\tF1_score: {round(f1,2)}")
        print(f"\tCoût métier: {round(cout_metier,2)}")

        mlflow.log_param('learning_rate', params['learning_rate'])
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_param('min_samples_split', params['min_samples_split'])

        # mlflow.sklearn.save_model(model, "GradientBoostingRegressor")

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('auc', auc_)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('coût_métier', cout_metier)


def mlflow_log_model(model, signature, artifact_path="Artifacts"):
    '''Logs a model via MLFlow.

    -------------
    Parameters:
    - model : sklearn model : Instance of a sklearn model among:
                                - sklearn.ensemble.RandomForestClassifier
                                - skleanr.ensemble.GradientBoostingClassifier
                                - sklearn.linear_model.LogisticRegression
    - signature : mlflow.ModelSignature : ModelSignature of the model to log.
    '''
    # Artifact logging
    mlflow.log_artifacts(artifact_path)

    # Model signature
    # signature = mlflow.models.signature.infer_signature(X_test, predictions)

    # Model logging
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        # registered_model_name="sk-learn-logistic-reg-model"
        registered_model_name="Credit_Score"
    )
    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")


# %%
