# %% Imports
import os

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px


# %% Functions
@st.cache_data
def load_clients_data():
    '''Charge les données de tous les clients disponibles.
    '''
    clients_data = pd.read_csv(os.path.join('Dataset', 'application_test.csv'))
    clients_data['NAME_INCOME_TYPE'].fillna('Non renseigne', inplace=True)
    clients_data['NAME_EDUCATION_TYPE'].fillna('Non renseigne', inplace=True)
    clients_data['NAME_FAMILY_STATUS'].fillna('Non renseigne', inplace=True)
    clients_data['OCCUPATION_TYPE'].fillna('Non renseigne', inplace=True)
    clients_data['OCCUPATION_TYPE'].replace(np.nan, "Non renseigne")

    return clients_data


@st.cache_data
def get_filtres():
    '''Crée les filtres sur la base des infos clients disponibles.
    '''
    filtres = {}

    all_data = pd.read_csv(os.path.join("Dataset", "application_test.csv"))
    all_data['OCCUPATION_TYPE'].replace(np.nan, "Non renseigne")
    filtres['type_revenu'] = all_data["NAME_INCOME_TYPE"].unique()
    filtres['education'] = all_data["NAME_EDUCATION_TYPE"].unique()
    filtres['family_status'] = all_data["NAME_FAMILY_STATUS"].unique()
    filtres['occupation'] = all_data["OCCUPATION_TYPE"].unique()
    for filtre_value in filtres['occupation']:
        if filtre_value is float:
            index = filtres['occupation'].index(filtre_value)
            filtres['occupation'][index] = "Non renseigne"
    filtres['occupation'][0] = "Non renseigne"

    return filtres


@st.cache_data
def get_client_data(client_id):
    '''Retourne les données spécifiques d'un client sous formes de dictionnaires.
    '''
    all_data = load_clients_data()
    client_data = all_data.query("SK_ID_CURR == @client_id")

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


def info_client(id):
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


def comparer_client(id):
    '''Compare le client aux autres clients.
    '''
    # Chargement des données de tous les clients pour comparaison
    all_clients_data = load_clients_data()
    dict_filtres = get_filtres()

    # Récupération des informations du client
    _, infos_fin, _ = get_client_data(id)

    data_graph = all_clients_data.query("AMT_INCOME_TOTAL < 500000 and AMT_ANNUITY.notna()")

    # Filtres sur Sidebar
    with st.sidebar.form(key='comparatif'):
        st.subheader("Filtres pour comparaison")

        filtre_revenu = st.multiselect('Source de revenu', dict_filtres['type_revenu'])  # , default=infos_aut['type_revenu'])
        filtre_education = st.multiselect('Education', dict_filtres['education'])  # , default=infos_aut['education'])
        filtre_statut_familial = st.multiselect('Statut familial', dict_filtres['family_status'])  # , default=infos_aut['family_status'])
        filtre_occupation = st.multiselect('Activité', dict_filtres['occupation'])  # , default=infos_aut['occupation'])

        submit_button_filtres = st.form_submit_button(label="Appliquer filtres")
    if submit_button_filtres:
        if len(filtre_revenu) > 0:
            data_graph = data_graph.query("NAME_INCOME_TYPE in @filtre_revenu")
        if len(filtre_education) > 0:
            data_graph = data_graph.query("NAME_EDUCATION_TYPE in @filtre_education")
        if len(filtre_statut_familial) > 0:
            data_graph = data_graph.query("NAME_FAMILY_STATUS in @filtre_statut_familial")
        if len(filtre_occupation) > 0:
            data_graph = data_graph.query("OCCUPATION_TYPE in @filtre_occupation")

    # Graphe des revenus
    NB_BINS = 25
    counts, bins = np.histogram(data_graph['AMT_INCOME_TOTAL'], bins=NB_BINS)
    bins_graph = bins + (bins[1]-bins[0])/2
    bins_graph = bins_graph[:-1]

    colors_graph = []
    for i in range(1, len(bins)):
        if infos_fin['revenus'] < bins[i] and infos_fin['revenus'] > bins[i-1]:
            colors_graph.append('client')
        else:
            colors_graph.append('')

    df_graph = pd.DataFrame([bins_graph, counts]).T
    df_graph.columns = ['Montants', 'Fréquence']

    fig1 = px.bar(df_graph,
                  x='Montants',
                  y='Fréquence',
                  color=colors_graph,
                  title="Distribution des revenus clients. Cappé à 500k€/an")
    fig1.update_layout(hovermode="closest")
    # fig1.update_layout(hovermode="<b>Fréquence: %{y:.2f}% <br>Revenu client:%{x}%")

    st.plotly_chart(fig1)

    # Graphe des annuités
    NB_BINS_ANNUITY = 25
    counts, bins = np.histogram(data_graph['AMT_ANNUITY'], bins=NB_BINS_ANNUITY)
    bins_graph = bins + (bins[1]-bins[0])/2
    bins_graph = bins_graph[:-1]

    colors_graph = []
    for i in range(1, len(bins)):
        if infos_fin['annuite'] < bins[i] and infos_fin['annuite'] > bins[i-1]:
            colors_graph.append('client')
        else:
            colors_graph.append('')

    df_graph = pd.DataFrame([bins_graph, counts]).T
    df_graph.columns = ['Montants', 'Fréquence']

    fig2 = px.bar(df_graph,
                  x='Montants',
                  y='Fréquence',
                  color=colors_graph,
                  title="Distribution des annuités annuelles clients.")
    # fig2.update_layout(hovermode="y unified")

    st.plotly_chart(fig2)

    # Graphe des encours crédits
    NB_BINS_CREDITS = 25
    counts, bins = np.histogram(data_graph['AMT_CREDIT'], bins=NB_BINS_CREDITS)
    bins_graph = bins + (bins[1]-bins[0])/2
    bins_graph = bins_graph[:-1]

    colors_graph = []
    for i in range(1, len(bins)):
        if infos_fin['encours_credit'] < bins[i] and infos_fin['encours_credit'] > bins[i-1]:
            colors_graph.append('client')
        else:
            colors_graph.append('')

    df_graph = pd.DataFrame([bins_graph, counts]).T
    df_graph.columns = ['Montants', 'Fréquence']

    fig3 = px.bar(df_graph,
                  x='Montants',
                  y='Fréquence',
                  color=colors_graph,
                  title="Distribution des encours crédits clients.")
    # fig3.update_layout(hovermode="y unified")

    st.plotly_chart(fig3)

    # Graphe des prix des biens
    NB_BINS_PRICE = 25
    counts, bins = np.histogram(data_graph['AMT_CREDIT'], bins=NB_BINS_PRICE)
    bins_graph = bins + (bins[1]-bins[0])/2
    bins_graph = bins_graph[:-1]

    colors_graph = []
    for i in range(1, len(bins)):
        if infos_fin['prix_biens'] < bins[i] and infos_fin['prix_biens'] > bins[i-1]:
            colors_graph.append('client')
        else:
            colors_graph.append('')

    df_graph = pd.DataFrame([bins_graph, counts]).T
    df_graph.columns = ['Montants', 'Fréquence']

    fig4 = px.bar(df_graph,
                  x='Montants',
                  y='Fréquence',
                  color=colors_graph,
                  title="Distribution des prix des biens.")
    # fig4.update_layout(hovermode="y unified")

    st.plotly_chart(fig4)


def app(id):
    '''Page
    '''
    st.title("COMPARATIF CLIENT")
    try:
        # Informations client
        info_client(id)
        st.write("*"*50)
        comparer_client(id)
    except IndexError:
        st.markdown(f""":red[Aucun client connu sous l'identifiant {id}.]""")

# %%
