# %%
from pickle import load
import pandas as pd

from fastapi import FastAPI

import mlflow
import mlflow.sklearn

import shap
import Data_prep


app = FastAPI()
MLFLOW_SERVING = "EC2"  # "Local" or "EC2" or "S3"


# %% Definitions

def read_file_to_list(filepath):
    '''Reads a file and return a list of its items.

    --------------
    Parameters:
        - filepath : str : Path to the file to read.
    Returns:
        - list_lue : list : List of elements read from the file.
    '''
    list_lue = []
    with open(filepath, 'r') as fp:
        for line in fp:
            x = line[:-1]
            list_lue.append(x)

    return list_lue


def get_user_data_scaled(id):
    '''Returns data related to an user, as formated in X_test.parquet file.

    --------------
    Parameters:
        - id : str : Client's SK_ID_CURR
    Returns:
        - DataFrame.row with user data, as formated in X_test.parquet file.
        - 0 if the user is not found.
        - else return a string indicating an error raised.
    '''
    x1, x2, _, _ = Data_prep.data_preparation(main_dataset_only=True, debug=False, new_data=True)

    new_data = pd.concat([x1, x2], axis=0)

    new_data.set_index(keys=['SK_ID_CURR'], drop=False, inplace=True)
    try:
        if id in new_data.index:
            # Récupération des données du client
            user_data = new_data.query('SK_ID_CURR == @id')

            imputer = load(open('imputer.pkl', 'rb'))

            # suppression des nouvelles features
            cols = read_file_to_list('cols.txt')
            x_train = pd.DataFrame(columns=cols)
            user_data, _ = user_data.align(x_train, join="right", axis=1)

            user_data = imputer.transform(user_data)
            user_data = pd.DataFrame(user_data, columns=cols)

            # Scaling des données pour le modèle
            scaler = load(open('scaler.pkl', 'rb'))
            x_new_scaled = scaler.transform(user_data.drop(columns='SK_ID_CURR'))
            cols.remove('SK_ID_CURR')
            x_new_scaled = pd.DataFrame(x_new_scaled, columns=cols)

            return x_new_scaled, user_data
        else:
            raise ValueError(f"id SK_ID_CURR {id} non reconnu")
    except ValueError as ve:
        print('In get_user_data_scaled/9')
        return f'ValueError while collecting data: {ve}'


# %% Instanciation du modèle
if MLFLOW_SERVING == "Local":
    model = mlflow.sklearn.load_model("models:/Credit_Score/Production")
if MLFLOW_SERVING == "EC2":
    model = load(open('model.pkl', 'rb'))
if MLFLOW_SERVING == "S3":
    model = mlflow.sklearn.load_model("s3://mlflow-creditscore-bucket/1/Production")


# %%  API
@app.get("/")
async def hello():
    '''Fonction d'accueil
    '''
    return {"Statut modèle": "Chargé"}


@app.get("/scoring/{SK_ID_CURR}")
async def scoring(SK_ID_CURR: int):
    '''Prédit le score d'un client sur la base de son id SK_ID_CURR
    '''
    user_data_scaled, user_data_unscaled = get_user_data_scaled(SK_ID_CURR)
    prediction_proba = model.predict_proba(user_data_scaled)
    # prediction = model.predict(user_data_scaled)

    features_name = user_data_scaled.columns
    explainer = shap.Explainer(model, user_data_scaled, feature_names=features_name)
    shap_values = explainer(user_data_scaled)

    # Local feature importance
    # print(shap_values[0].base_values)
    dict_shap = {'base_value': shap_values[0].base_values}
    NB_VALUES = 10
    df_shap = pd.DataFrame(shap_values[0].data)
    df_shap.index = features_name
    df_shap.columns = ['values']
    df_shap['abs'] = abs(df_shap['values'])
    df_shap.sort_values(by=['abs'], ascending=False, inplace=True)
    df_shap = df_shap.head(NB_VALUES)
    print("df_shap.index:", df_shap.index)
    # print("user_data_unscaled:", user_data_unscaled)
    df_true_value = user_data_unscaled.drop(columns=[col for col in user_data_scaled if col not in df_shap.index])
    df_true_value = df_true_value.T
    df_true_value.columns = ['true_values']
    df_true_value['feature'] = df_true_value.index
    print("df_true_value:", df_true_value)

    df_shap['feature'] = df_shap.index
    dict_series = df_shap.to_dict('series')
    dict_values = dict(zip(dict_series['feature'], dict_series['values']))
    dict_shap['values'] = dict_values
    print("df_shap:", df_shap)
    print("dict_shap:", dict_shap)

    dict_series = df_true_value.to_dict('series')
    dict_true_values = dict(zip(dict_series['feature'], dict_series['true_values']))
    dict_shap['true_values'] = dict_true_values

    if prediction_proba[0][0] > .5:
        pass_fail = 'pass'
    else:
        pass_fail = "fail"
    return {"Client_ID": SK_ID_CURR,
            # "Probabilité": round(max(prediction_proba[0]), 4),
            "Probabilité": round(prediction_proba[0][0], 4),
            "Pass": pass_fail,
            "shap": dict_shap}


# %%
if __name__ == "__main__":
    pass
# %%
