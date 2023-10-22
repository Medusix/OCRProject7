# %%
import pandas as pd

from pickle import load

from fastapi import FastAPI

import mlflow
import mlflow.sklearn
import Data_prep

app = FastAPI()


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
        return "SK_ID_CURR non présent en base"
    except ValueError as ve:
        return f'ValueError while collecting data. {ve}'


# %% Instanciation du modèle
model = mlflow.sklearn.load_model("models:/Credit_Score/Production")


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
    print('In /scoring/')
    user_data_scaled = get_user_data_scaled(SK_ID_CURR)
    prediction = model.predict(user_data_scaled)
    if prediction < .5:
        result = 'Pass'
    else:
        result = 'Fail'
    return {"Score": result}


# %%
if __name__ == "__main__":
    pass
# %%
