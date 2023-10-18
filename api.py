# %%
import pandas as pd
import os

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

app = FastAPI()


# %% Definitions
def get_user_data(id):
    '''Returns data related to an user, as formated in X_test.parquet file.

    --------------
    Parameters:
        - id : str : Client's SK_ID_CURR
    Returns:
        - DataFrame.row with user data, as formated in X_test.parquet file.
        - 0 if the user is not found.
        - else return a string indicating an error raised.
    '''
    X_test = pd.read_parquet(os.path.join("Dataset", "X_test.parquet"))
    try:
        if id in X_test['SK_ID_CURR']:
            return X_test.query('SK_ID_CURR == @id')
        return 0
    except Exception:
        return 'Error while collecting data.'


# %% Instanciation du modèle
# 1. Récupérer le modèle
# mlflow.load_model (nom modèle / prod) etc.
model = mlflow.sklearn.load_model("models:/Credit_Score/Production")
model.predict()



# %%  API
@app.get("/")
async def scoring():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# %%
if __name__ == "__main__":
    pass
# %%
