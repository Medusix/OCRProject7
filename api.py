# %%
from pickle import load
import pandas as pd


from fastapi import FastAPI

import mlflow
import mlflow.sklearn
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
    print('In get_user_data_scaled/1')
    x1, x2, _, _ = Data_prep.data_preparation(main_dataset_only=False, debug=False, new_data=True)
    print('In get_user_data_scaled/1.2')
    # print('x1.head(2):')
    # print(x1.head(2))
    # print('x2.head(2):')
    # print(x2.head(2))
    new_data = pd.concat([x1, x2], axis=0)
    # print("new_data.head():")
    # print(new_data.head())
    new_data.set_index(keys=['SK_ID_CURR'], drop=False, inplace=True)
    print('In get_user_data_scaled/2')
    try:
        if id in new_data.index:
            print('In get_user_data_scaled/3')
            # Récupération des données du client
            user_data = new_data.query('SK_ID_CURR == @id')
            print('In get_user_data_scaled/4')

            imputer = load(open('imputer.pkl', 'rb'))
            print('In get_user_data_scaled/5')

            # suppression des nouvelles features
            cols = read_file_to_list('cols.txt')
            x_train = pd.DataFrame(columns=cols)
            # x_train, _, _, _ = Data_prep.data_preparation()
            user_data, _ = user_data.align(x_train, join="right", axis=1)
            # user_data = user_data[user_data.columns.intersection(cols)]
            print('In get_user_data_scaled/4')

            user_data = imputer.transform(user_data)
            # user_data = pd.DataFrame(user_data, columns=x_train.columns)
            user_data = pd.DataFrame(user_data, columns=cols)
            print('In get_user_data_scaled/5')

            # Scaling des données pour le modèle
            scaler = load(open('scaler.pkl', 'rb'))
            print('In get_user_data_scaled/6')
            x_new_scaled = scaler.transform(user_data.drop(columns='SK_ID_CURR'))
            # x_new_scaled = pd.DataFrame(x_new_scaled, columns=x_train.drop(columns='SK_ID_CURR').columns)
            cols.remove('SK_ID_CURR')
            x_new_scaled = pd.DataFrame(x_new_scaled, columns=cols)

            print('In get_user_data_scaled/7')
            return x_new_scaled
        else:
            print('In get_user_data_scaled/8')
            raise ValueError(f"id SK_ID_CURR {id} non reconnu")
    except ValueError as ve:
        print('In get_user_data_scaled/9')
        return f'ValueError while collecting data. {ve}'


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
    print('In /scoring/1')
    user_data_scaled = get_user_data_scaled(SK_ID_CURR)
    print('In /scoring/2')
    prediction_proba = model.predict_proba(user_data_scaled)
    print('In /scoring/3')
    prediction = model.predict(user_data_scaled)
    print('In /scoring/4')
    if prediction < .5:
        pass_fail = 'pass'
    else:
        pass_fail = "fail"
    print('In /scoring/5')
    print(prediction)
    return {"Client_ID": SK_ID_CURR,
            # "Probabilité": round(max(prediction_proba[0]), 4),
            "Probabilité": round(prediction_proba[0][0], 4),
            "Pass": pass_fail}


# %%
if __name__ == "__main__":
    pass
# %%
