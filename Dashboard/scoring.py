# %% Imports
import os
import streamlit as st
import json
import requests
import pandas as pd
import plotly.express as px


# %% Functions
def build_shap_graph(dict_values, dict_true_values, exclure_base_value=True):
    '''Construit et retourne un graphe présentant les valeurs d'influence principales.
    '''
    dict_shap = dict_values.copy()
    for key in dict_shap:
        dict_shap[key] *= -1
    dict_value_display = dict_true_values
    if exclure_base_value:
        dict_shap.pop('base_value')
        dict_value_display.pop('base_value')
    colors_shap = []
    for value in dict_shap.values():
        if value < 0:
            colors_shap.append("red")
        else:
            colors_shap.append("green")
    value_display = []
    for value in dict_value_display.values():
        value_display.append(value)

    fig = px.bar(
        x=dict_shap.keys(),
        y=dict_shap.values(),
        color=colors_shap,  # color=value_display,
        labels=["Variables", "Valeurs", 'indicateur'],
        title="Principaux facteurs (valeurs relatives au modèle)"
        )
    # fig.update_layout(hovermode="y unified")

    return fig


@st.cache_data
def load_clients_data():
    '''Charge les données de tous les clients disponibles.
    '''
    clients_data = pd.read_csv(os.path.join('Dataset', 'application_test.csv'))
    clients_data['NAME_INCOME_TYPE'].fillna('Non renseigne', inplace=True)
    clients_data['NAME_EDUCATION_TYPE'].fillna('Non renseigne', inplace=True)
    clients_data['NAME_FAMILY_STATUS'].fillna('Non renseigne', inplace=True)
    clients_data['OCCUPATION_TYPE'].fillna('Non renseigne', inplace=True)
    return clients_data


@st.cache_data
def get_client_data(client_id):
    '''Retourne les données spécifiques d'un client sous formes de dictionnaires.
    '''
    # all_data = pd.read_csv(os.path.join("Dataset", "application_test.csv"))
    all_data = load_clients_data()
    client_data = all_data.query("SK_ID_CURR == @client_id")
    # client_data['NAME_INCOME_TYPE'].fillna('Non renseigné', inplace=True)
    # client_data['NAME_EDUCATION_TYPE'].fillna('Non renseigné', inplace=True)
    # client_data['NAME_FAMILY_STATUS'].fillna('Non renseigné', inplace=True)
    # client_data['OCCUPATION_TYPE'].fillna('Non renseigné', inplace=True)
    raw_id = client_data.index
    infos_gen = {}
    infos_fin = {}
    infos_aut = {}

    # Informations générales
    sexe = 'Homme' if client_data['CODE_GENDER'][raw_id].values[0] == 'M' else 'Femme'
    infos_gen['id'] = client_id
    infos_gen['Sexe'] = sexe
    infos_gen['statut_familial'] = client_data['NAME_FAMILY_STATUS'][raw_id].values[0]
    infos_gen['nb_enfants'] = client_data['CNT_CHILDREN'][raw_id].values[0]

    # Informations financières
    infos_fin['revenus'] = client_data['AMT_INCOME_TOTAL'][raw_id].values[0]
    infos_fin['encours_credit'] = client_data['AMT_CREDIT'][raw_id].values[0]
    infos_fin['annuite'] = client_data['AMT_ANNUITY'][raw_id].values[0]
    infos_fin['prix_biens'] = client_data['AMT_GOODS_PRICE'][raw_id].values[0]

    # Informations autres
    infos_aut['type_revenu'] = client_data['NAME_INCOME_TYPE'][raw_id].values[0]
    infos_aut['education'] = client_data['NAME_EDUCATION_TYPE'][raw_id].values[0]
    infos_aut['family_status'] = client_data['NAME_FAMILY_STATUS'][raw_id].values[0]
    infos_aut['occupation'] = client_data['OCCUPATION_TYPE'][raw_id].values[0]

    return infos_gen, infos_fin, infos_aut


def get_client_scoring(SK_ID_CURR, hosting='cloud'):
    '''Prédit
    '''
    if hosting == 'local':
        url = f"http://127.0.0.1:8000/scoring/{SK_ID_CURR}"
    if hosting == 'cloud':
        url = f"http://13.49.44.23:5001/scoring/{SK_ID_CURR}"
    response = requests.get(url)
    dict_result = json.loads(response.text)

    return dict_result


# @st.cache_data
def predict(id):
    ''' Prédit
    '''
    results = get_client_scoring(SK_ID_CURR=id)
    # results = get_client_scoring(SK_ID_CURR=id, hosting='local')
    print(results)

    if results["Pass"] == "pass":
        st.success(f"**Client éligible à un crédit. Probabilité de remboursement de son crédit dans les temps: {results['Probabilité']*100:.2f}%**")
    else:
        st.error(f"**Client non éligible à un crédit. Probabilité de remboursement dans les temps: {results['Probabilité']*100:.2f}%**")

    # st.text(f"Probabilité de remboursement de son crédit dans les temps: {results['Probabilité']*100:.2f}%")

    dict_values = {"base_value": results['shap']['base_value']}
    dict_values.update(results['shap']['values'])
    dict_true_values = {"base_value": 0}
    dict_true_values.update(results['shap']['true_values'])
    fig_shap = build_shap_graph(dict_values, dict_true_values)
    st.plotly_chart(fig_shap)

    # Affichage des valeurs brutes
    dict_display = print_key_values(dict_true_values)
    st.write("Valeurs réelles")
    st.write(dict_display)
    # Affichage des valeurs SHAP
    st.write("Valeurs SHAP")
    st.write(dict_values)


def print_key_values(dict_values):
    dict_values_display = dict_values.copy()
    for (key, value) in dict_values.items():
        if key == "CODE_GENDER":
            print("In print_key_values / CODE_GENDER")
            if value == 0:
                dict_values_display[key] = "Femme"
            else:
                dict_values_display[key] = "Homme"
        if key in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_EMAIL"]:
            if value == 0:
                dict_values_display[key] = "Non"
            else:
                dict_values_display[key] = "Oui"
        if key in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', "AMT_GOODS_PRICE"]:
            dict_values_display[key] = str(value) + "€"
        if key in ['DAYS_BIRTH', 'DAYS_EMPLOYED']:
            dict_values_display[key] = str(round(-value//365, 0)) + " ans"
        if key in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE', 'DAYS_EMPLOYED_PERC']:
            dict_values_display[key] = round(value, 2)

    return dict_values_display


def info_clients(id):
    '''Affiche les information du client.
    '''
    # Récupération des informations du client
    infos_gen, infos_fin, infos_aut = get_client_data(id)

    # Informations clients
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informations générales")
        st.write("Client id:", infos_gen['id'])
        st.write("Sexe:", infos_gen['Sexe'])
        st.write("Statut familial:", infos_gen['statut_familial'])
        st.write('Nombre d\'enfants:', infos_gen['nb_enfants'])
    with col2:
        st.subheader("Informations financières")
        col21, col22 = st.columns(2)
        with col21:
            st.write("Revenus annuels:")
            st.write("Encours crédits:")
            st.write("Annuités:")
            st.write("Valeurs biens:")
        with col22:
            st.write(f"{infos_fin['revenus']:.2f}€")
            st.write(f"{infos_fin['encours_credit']:.2f}€")
            st.write(f"{infos_fin['annuite']:.2f}€")
            st.write(f"{infos_fin['prix_biens']:.2f}€")


def app(id):
    '''Page'''
    st.title("SCORING CLIENT")
    try:
        info_clients(id)
        st.write("*"*50)
        predict(id)
    except IndexError:
        st.markdown(f""":red[Aucun client connu sous l'identifiant {id}.]""")

# %%
