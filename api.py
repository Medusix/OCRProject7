# %%
from flask import jsonify, Flask, render_template
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

import json
import requests

# %% Instanciation de l'API
# app = Flask(__name__)
app = FastAPI()

class client_data(BaseModel):
    


# %% Instanciation du modèle

# 1. Récupérer le modèle
# mlflow.load_model (nom modèle / prod) etc.


# %%  API
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/credit_score/{}")

# %%
if __name__ == "__main__":
    pass
# %%
