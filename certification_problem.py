import torch
import torch.nn as nn
import torch.nn.functional as F

import abc
from reseau_train import Reseau, architectures_modele


from comparaison_modeles_outils import(
    load_data,
    retourne_weights
)


class Certification_Problem:
    def __init__(
            self,
            data_modele : str = "MNIST",
            architecture : str = None,
            epsilon : float = 0.1):
        
        print("Initialisation du problème de certification...")
        n, K = architectures_modele(data_modele,architecture)
        file, data = load_data(data_modele, architecture)
        W, b = retourne_weights(K, n, file)
        self.n, self.K, self.W, self.b = n, K, W, b
        self.data = data
        self.Res = Reseau(K, n, W, b)
        self.epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon
    
    @property
    def Res(self):
        return self._Res
    
    @property
    def K(self):
        return self._K
    
    @property
    def n(self):
        return self._n

    @property   
    def W(self):
        return self._W
    
    @property
    def b(self):
        return self._b
    
    @epsilon.setter  # setter parce que epsilon est une propriété
    def epsilon(self, value):
        if value < 0:
            raise ValueError("epsilon doit être positif")
        self._epsilon = value

    @Res.setter 
    def Res(self, value):
        if not isinstance(value, Reseau):
            raise ValueError("Res doit être une instance de Reseau")
        self._Res = value

    @K.setter   
    def K(self, value):
        if not isinstance(value, int):
            raise ValueError("K doit être un entier")
        self._K = value

    @n.setter
    def n(self, value):
        if not isinstance(value, list):
            raise ValueError("n doit être une liste")
        self._n = value

    @W.setter
    def W(self, value):
        if not isinstance(value, list):
            raise ValueError("W doit être une liste")
        self._W = value

    @b.setter
    def b(self, value):
        if not isinstance(value, list):
            raise ValueError("b doit être une liste")
        self._b = value
    
    
    
    





