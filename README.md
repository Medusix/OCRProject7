# OCRProject7
Repository d'un projet de scoring crédit réalisé par Rémi Vaillant dans le cadre du projet 7 de la formation Data Scientist d'OpenClassRoom.

PROJET

Contexte:
La société financière Prêt à Dépenser propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d’historique de prêt. La société souhaite mettre en œuvre un outil de « scoring crédit » pour calculer la probabilité qu’un client rembourse son crédit.
Objectif :
1.	 Développer un modèle de prédiction probabilité de faillite d'un client.
2.	 Analyser les features qui contribuent le plus au modèle.
3.	 Mettre en production le modèle de prédiction à l’aide d’une API.
4.	Mettre en oeuvre une approche globale MLOps de bout en bout


COMPOSITION DU DOSSIER
Le dossier principal comprend les codes sources nécessaire au déroulé de bout en bout du projet:
- data_prep.py: code de préparation des données
- modelisations.py: code des différentes modélisations.
- mlflow_tools.py: toolkit de fonctions utilisées pour enregistrer différents composants MLFlow (experiments, models, scores, artefacts)
- analyse_shap.py: code d'analyse de feature importance (global et local)
- api.py: code de l'API pour (nécessite une mise en server eg. uvicorn)
- Test appel api de scoring.ipynb: Notebook de test d'un appel d'API
- tests_unitaires.py: code pour l'exécution automatisée de tests unitaires lors du build Github.
- Divers fichiers pickle utilisés dans le pipeline de modélisation (imputer.pkl, model.pkl, scaler.pkl)
à l'exécution de l'API, en passant par la modélisation et les différentes éléments de pipelines
- requirements.txt: liste des packages python requis.

Trois autres dossiers sont présents:
- .github/workflows: contient le fichier blank.yml gérant le build github actions.
- mlruns: contient les runs mlrun locaux.
- Dataset: les données du projets.