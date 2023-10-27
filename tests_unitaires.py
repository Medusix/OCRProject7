# %%
'''Fichier de tests unitaires pour exécution pytest lors du build Github.
'''
import numpy as np
from modelisations import addition, cout_metier


def test_addition():
    '''Assertion de la fonction addition.
    
    '''
    assert addition() == 0
    assert addition(1, 2) == 3
    assert addition(2, 2) == 4


def test_cout_metier():
    '''Assertion de la fonction cout_metier.
    
    '''
    y_t = np.asarray([0, 0, 0, 1, 1])
    y_p1 = np.asarray([0, 0, 0, 1, 1])
    y_p2 = np.asarray([1, 1, 1, 0, 0])

    assert cout_metier(y_t, y_p1) == 0
    assert cout_metier(y_t, y_p2) == 4.6


# %%
