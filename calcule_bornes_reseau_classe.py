import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import abc

dossier_gurobi = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models_gurobi')
sys.path.append(dossier_gurobi)

from Models_gurobi.GUROBI_outils import parametres_gurobi

from Models_gurobi.GUROBI_outils import (
    adapt_parametres_gurobi, 
    retourne_valeurs_solutions_bornes
)

from Models_gurobi.GUROBI_objectif import (
    add_objective_U,
    add_objective_L,
    add_objective_L_linear,
    add_objective_U_linear
)
from Models_gurobi.GUROBI_variables import (
    add_variable_z,
    add_variable_sigma
)
from Models_gurobi.GUROBI_contraintes import (
    add_initial_ball_constraints,
    add_hidden_layer_constraints_with_sigma_linear_Glover,
    add_last_layer
)
from certification_problem import Certification_Problem


class Bornes(abc.ABC):
    def __init__(self, 
                 probleme_certification : Certification_Problem):
        self.pb_cert = probleme_certification


    def IBP(self,
            x0:torch.Tensor):
        L = [[(x0[j]-self.pb_cert.epsilon) for j in range(self.pb_cert.n[0])]]
        U = [[(x0[j]+self.pb_cert.epsilon) for j in range(self.pb_cert.n[0])]]

        neurones_actifs_stables = []
        neurones_inactifs_stables = []


        for idx,layer in enumerate(self.pb_cert.Res.layers):
            if isinstance(layer, nn.Linear):  # Vérifie si c'est une couche linéaire
                weight_layer = layer.weight.detach().clone()  # Récupère les poids
                 # Change L_layer pour le dupliquer pour chaque neurone
                print("ind : ", idx)
                L_previous_layer = torch.tensor(L[idx//2]).view(-1, 1).repeat(1, layer.out_features)   
                
                U_previous_layer = torch.tensor(L[idx//2]).view(-1, 1).repeat(1, layer.out_features)   
                print(f"Shape L_previous_layer : {L_previous_layer.shape}")
                print(f"Shape U_previous_layer : {U_previous_layer.shape}")
                print(f"Shape weight_layer : {weight_layer.shape}")
                print(f"Shape weight apres masque : {torch.where(weight_layer > 0, weight_layer, torch.tensor(0.0)) .shape}")

                L_layer = torch.sum(L_previous_layer.T * torch.where(weight_layer > 0, weight_layer, torch.tensor(0.0)) 
                                    + U_previous_layer.T * torch.where(weight_layer < 0, weight_layer, torch.tensor(0.0))  , 
                                    dim=1)  # Calcule la borne inférieure de la couche
                U_layer = torch.sum(U_previous_layer.T * torch.where(weight_layer > 0, weight_layer, torch.tensor(0.0)) 
                                    + L_previous_layer.T * torch.where(weight_layer < 0, weight_layer, torch.tensor(0.0))  , 
                                    dim=1)  # Calcule la borne supérieure de la couche
                print(f"Shape L_layer : {L_layer.shape}")
                print(f"Shape U_layer : {U_layer.shape}")
                L.append(L_layer.tolist())
                U.append(U_layer.tolist())

                if (idx//2+1) < self.pb_cert.K:
                    neurones_actifs_couche = torch.nonzero(L_layer>0).squeeze()
                    neurones_inactifs_couche = torch.nonzero(U_layer<0).squeeze()

                    neurones_actifs_stables.append([(idx//2+1, ind.item()) for ind in neurones_actifs_couche])
                    neurones_inactifs_stables.append([(idx//2+1, ind.item()) for ind in neurones_inactifs_couche])
            
        print("Resultats de l'IBP : ")
        print(f"L = {L[1:]}")
        print(f"U = {U[1:]}")
        print("Neurones actifs stables : ", neurones_actifs_stables)
        print("Neurones inactifs stables : ", neurones_inactifs_stables)
        return L, U, neurones_actifs_stables, neurones_inactifs_stables
    



if __name__ == "__main__":
    data_modele = "MNIST"
    architecture = "2x20"
    epsilon = 0.1
    Certification_Problem_MNIST = Certification_Problem(data_modele, architecture, epsilon)
    x0, y0 = Certification_Problem_MNIST.data[0][0][0], Certification_Problem_MNIST.data[0][0][1]
    print("x0 : ", x0)
    x0 = np.array(x0).reshape(784).tolist()  
    bornes = Bornes(Certification_Problem_MNIST)
    bornes.IBP(x0)