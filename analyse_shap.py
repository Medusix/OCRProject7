# %%
from pickle import load
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import shap

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

# %%
if __name__ == '__main__':
    model = load(open('model.pkl', 'rb'))
    
    features = read_file_to_list('cols.txt')
    
    if isinstance(model, LogisticRegression):
        # explainer = shap.Explainer(model, X_test)
        # shap_values = explainer(X_test)
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_train_scaled)

        shap.plots.beeswarm(shap_values)
        
        for user in range(0,10):
            shap.force_plot(base_value = explainer.expected_value, shap_values = shap_values.values[user])

    if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(model)
        explainer.expected_value = explainer.expected_value[0]
        shap_values = explainer.shap_values(X_test_scaled)

        newCmap = LinearSegmentedColormap.from_list("", ['#c4cfd4', '#3345ea'])
        cmap = plt.get_cmap('tab10')
        MAX_FEAT_DISPLAY = 14
        plt.title(f"Contributions des {MAX_FEAT_DISPLAY} principales features au score global", fontsize=14)
        # shap.summary_plot(shap_values, X_train_scaled, plot_type="violin", color=newCmap, max_display=10)
        # shap.summary_plot(shap_values, X_train_scaled, plot_type="violin", max_display=10)
        shap.summary_plot(
            shap_values,
            features=features,
            max_display=MAX_FEAT_DISPLAY,
            plot_type='bar',  # 'violin' ou 'dot' ou 'bar'
            plot_size=(12, 6))
# %%
