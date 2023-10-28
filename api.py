# %%
import pandas as pd

from pickle import load
from pydantic import BaseModel

from fastapi import FastAPI

import mlflow
import mlflow.sklearn
import Data_prep

app = FastAPI()
MLFLOW_SERVING = "EC2"  # "Local" or "EC2" or "S3"


class Score(BaseModel):
    '''Objet de retour de l'API comprenant différentes informations.
    '''
    client: int
    pass_fail: str
    score: float


# %% Definitions
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
            x_train, _, _, _ = Data_prep.data_preparation()
            user_data, _ = user_data.align(x_train, join="right", axis=1)

            user_data = imputer.transform(user_data)
            user_data = pd.DataFrame(user_data, columns=x_train.columns)

            # Scaling des données pour le modèle
            scaler = load(open('scaler.pkl', 'rb'))
            x_new_scaled = scaler.transform(user_data.drop(columns='SK_ID_CURR'))
            x_new_scaled = pd.DataFrame(x_new_scaled, columns=x_train.drop(columns='SK_ID_CURR').columns)

            return x_new_scaled
        else:
            raise ValueError(f"id SK_ID_CURR {id} non reconnu")
    except ValueError as ve:
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
    return {"Chargement": "Modèle"}


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
            "Probabilité": round(max(prediction_proba[0]), 4),
            "Pass": pass_fail}

    '''user_score = Score({"client": SK_ID_CURR,
                        "pass_fail": pass_fail,
                        "score": prediction
                        })
    return user_score'''


# %%
if __name__ == "__main__":
    pass
# %%
