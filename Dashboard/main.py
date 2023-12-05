import streamlit as st
from streamlit_option_menu import option_menu
import scoring
import comp
import accueil


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run():
        '''Fonction run
        '''
        with st.sidebar:
            st.image("images/logo.png")

            app = option_menu(
                menu_title="Menu",
                options=['Accueil', 'Scoring', 'Comparatif'],
                icons=['house-fill', 'trophy-fill', 'person-circle'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {'padding': '5!important', 'background-color': 'white'},
                    "icon": {"color": 'black', 'font-size': "18px"},
                    "nav-link": {"color": "black", "font-size": "15px", "text-align": 'left'},
                    "nav-link-selected": {"background-color": "#02ab21"}
                    }
            )

            st.header("Sélection client")
            client_id = st.number_input('Numéro SK_ID_CURR', step=1, value=100001)

        if app == 'Accueil':
            accueil.app(client_id)

        if app == 'Scoring':
            scoring.app(client_id)

        if app == 'Comparatif':
            comp.app(client_id)

    run()
