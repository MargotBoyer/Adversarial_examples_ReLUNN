import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import itertools

from load import load_data, load_file_weights, retourne_weights, cherche_ycible
from reseau_train import Reseau, architectures_modele

from calcule_bornes_reseau import (
    solve_borne_inf_L,
    solve_borne_sup_U
)
from comparaison_modeles_outils import compute_adverse
from Models_gurobi.GUROBI_outils import parametres_gurobi

import Models_MOSEK.Model_FprG_d as MOSEK_FprG_d
import Models_MOSEK.Model_FprG as MOSEK_FprG
import Models_MOSEK.Model_Lan_couches as MOSEK_Lan_couches
import Models_MOSEK.Model_Lan as MOSEK_Lan
import Models_MOSEK.Model_Mix_couches as MOSEK_Mix_couches


optimization_models_lineaires = ["Glover_Obj_diff","Fischetti_Obj_diff","Fischetti_Obj_dist","ReLUconvexe_Adv1"]
optimization_models_quadratiques = ["Mix","FprG_quad","Mix_diff_obj_quad","Adv2_ReLU2_lin","F_ReLU1_Adv2",
                                    "ReLU2_Adv2__FprGdiff__","Adv3_ReLU2_lin","F_ReLU1_Adv3","Adv3_ReLU2_lin",
                                    "ReLU3_Adv3","Adv3_ReLU2","F_ReLU1_Adv3","Lan_quad"]
optimization_models_gurobi = optimization_models_lineaires + optimization_models_quadratiques
optimization_models_mosek = [ "Mix_SDP","Mix_couches_SDP","Mix_d_SDP","FprG_SDP","Mix_d_couches_SDP","Lan_SDP","Lan_couches_SDP"]




class DatasetCertification(Dataset):
    def __init__(self, data_modele, architecture):
        file, data = load_data(data_modele, architecture)
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index][0], self.data[0][index][1]

    def __len__(self):
        return len(self.data[0])



class Certification_Problem_Data:
    def __init__(
            self,
            data_modele : str, 
            architecture : str = None,
            x0 : torch.Tensor = None,
            y0 : int = None,
            x0_id : int = 0,
            epsilon : float = 0.1):
        
        print(f"Initialisation du problème pour la donnée d'indexe {x0_id}")
        n, K = architectures_modele(data_modele,architecture)
        file = load_file_weights(data_modele, architecture)
        W, b = retourne_weights(K, n, file)
        self.n, self.K, self.W, self.b = n, K, W, b
        self.Res = Reseau.create_with_file(data_modele, architecture)
        self.epsilon = epsilon
        self.L = None
        self.U = None
        self.W_reverse = [[ [couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))] for couche in W]
        self.x0 = x0
        self.x0_id = x0_id
        self.y0 = y0
        self.resultats = []
        self.data_modele = data_modele
        self.optimization_models = []
        self.neurones_actifs_stables = []
        self.neurones_inactifs_stables = []
        self.rho = 0

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
    
    @property
    def L(self):
        return self._L
    
    @property
    def U(self):
        return self._U  
    
    @property
    def x0(self):
        return self._x0
    
    @property
    def y0(self):
        return self._y0
    
    @property
    def rho(self):
        return self._rho
    
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

    @L.setter
    def L(self, value):
        if not isinstance(value, list) and value is not None:
            raise ValueError("L doit être une liste")
        self._L = value

    @U.setter
    def U(self, value): 
        if not isinstance(value, list) and value is not None:
            raise ValueError("U doit être une liste")
        self._U = value


    @x0.setter
    def x0(self, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("x0 doit être un torch.Tensor")
        self._x0 = value

    @y0.setter
    def y0(self, value):
        if not isinstance(value, int) and not isinstance(value,float):
            print("Value : ", value)
            raise ValueError("y0 doit être un int ou un float")
        self._y0 = value

    @rho.setter
    def rho(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError("rho doit etre un int ou float")
        self._rho = value
    #Ecrire une méthode qui récupère une donnée à tester


    def IBP(self):
        """Calcule les bornes inférieures et supérieures des activations des neurones pour une entrée donnée

        Args:
            x0 (torch.Tensor): entrée à tester

        Returns:
            _type_: Bornes inférieures et supérieures des activations des neurones et neurones actifs et inactifs stables
        """
        print(f"x0 : {self.x0}")
        L = [[(self.x0[j].item()-self.epsilon) for j in range(self.n[0])]]
        U = [[(self.x0[j].item()+self.epsilon) for j in range(self.n[0])]]

        neurones_actifs_stables = []
        neurones_inactifs_stables = []


        for idx,layer in enumerate(self.Res.layers):
            if isinstance(layer, nn.Linear):  # Vérifie si c'est une couche linéaire
                weight_layer = layer.weight.detach().clone()  # Récupère les poids
                 # Change L_layer pour le dupliquer pour chaque neurone
                print("ind : ", idx)
                L_previous_layer = torch.tensor(L[idx//2]).view(-1, 1).repeat(1, layer.out_features)   
                
                U_previous_layer = torch.tensor(U[idx//2]).view(-1, 1).repeat(1, layer.out_features)   
                # print(f"Shape L_previous_layer : {L_previous_layer.shape}")
                # print(f"Shape U_previous_layer : {U_previous_layer.shape}")
                print(f"Previous layer borne inf L : {L_previous_layer}")
                print(f"Previous layer borne inf U : {L_previous_layer}")
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

                if (idx//2+1) < self.K:
                    neurones_actifs_couche = torch.nonzero(L_layer>0).view(-1)
                    neurones_inactifs_couche = torch.nonzero(U_layer<0).view(-1)

                    print(f"Idx : {idx//2+1}, neurones actifs : {neurones_actifs_couche}, neurones inactifs : {neurones_inactifs_couche}")
                    neurones_actifs_stables.extend([(idx//2+1, ind.item()) for ind in neurones_actifs_couche])
                    neurones_inactifs_stables.extend([(idx//2+1, ind.item()) for ind in neurones_inactifs_couche])
                    print(f"Neurones actifs  la fin de la couche {idx//2+1} : {neurones_actifs_stables}")
                    print(f"Neurones inactifs  la fin de la couche {idx//2+1} : {neurones_inactifs_stables}")
                    print()
        print("Resultats de l'IBP : ")
        print(f"L = {L}")
        print(f"U = {U}")
        self.L = L
        self.U = U
        self.neurones_actifs_stables = neurones_actifs_stables
        self.neurones_inactifs_stables = neurones_inactifs_stables
        return L, U, neurones_actifs_stables, neurones_inactifs_stables    



    def compute_FULL_U_L(self, relax = False,
        verbose = False, neurones_actifs_stables = [], neurones_inactifs_stables = []):
        """Calcule les bornes inférieures et supérieures des activations des neurones par des modèles MILP (exacts)

        Args:
            x0 (_type_): entrée à tester
            L (_type_): borne inf déjà connue (mais pas forcément exacte)
            U (_type_): borne sup déjà connue (mais pas forcément exacte)
            verbose (bool, optional): _description_. Defaults to False.
            neurones_actifs_stables (list, optional): neurones actifs stables déjà connus. 
            neurones_inactifs_stables (list, optional): neurones inactifs stables déjà connus.

        Returns:
            Bornes inférieures et supérieures exactes, et ensembles de neurones actifs et inactifs stables exacts
        """
        L_new = []
        U_new = []

        parametres_gurobi["DualReduction"] = 0
        parametres_gurobi.update({"TimeLimit" : 20})
        L_new.append([self.x0[feature].item() - self.epsilon   for feature in range(self.n[0])])
        U_new.append([self.x0[feature].item() + self.epsilon   for feature in range(self.n[0])])

        for couche in range(1,self.K+1):
            L_couche= []
            U_couche= []
            if couche== self.K: 
                verbose = True
            for neurone in range(self.n[couche]):
                if (couche,neurone) in neurones_actifs_stables+ neurones_inactifs_stables:
                    print(f"Le neurone {neurone} de la couche {couche} n'est pas traite car stable.")
                    L_couche.append(self.L[couche][neurone])
                    U_couche.append(self.U[couche][neurone])
                    continue
                print(f"\n     Couche {couche}  neurone {neurone}  : ")
                print("Neurones actifs stables a ce jour : ", neurones_actifs_stables)
                print("Neurones inactifs stables a ce jour : ", neurones_inactifs_stables)
                L_neurone, status, time_execution, dic_nb_nodes = solve_borne_inf_L(couche, neurone, self.K, self.n, self.x0,  self.W_reverse, self.b, self.L, self.U, self.epsilon, 
                                                                                    relax, neurones_actifs_stables, neurones_inactifs_stables, parametres_gurobi, verbose = verbose)            
                L_couche.append(L_neurone)
                print(f"L = {L_neurone}")

                U_neurone, status, time_execution, dic_nb_nodes = solve_borne_sup_U(couche, neurone, self.K, self.n, self.x0,  self.W_reverse, self.b, self.L, self.U, self.epsilon, 
                                                                                    relax, neurones_actifs_stables, neurones_inactifs_stables, parametres_gurobi,verbose = verbose)
                print(f"U = {U_neurone}")
                U_couche.append(U_neurone)

                if (couche>0) & (couche<self.K) & (U_neurone < 0):
                    print(f"Le neurone {neurone} de la couche {couche} est inactif stable.")
                    neurones_inactifs_stables.append((couche,neurone))
                elif (couche>0) & (couche<self.K) & (L_neurone > 0) : 
                    neurones_actifs_stables.append((couche,neurone))
                    print(f"Le neurone {neurone} de la couche {couche} est actif stable.")
            L_new.append(L_couche)
            U_new.append(U_couche)

        print("Les neurones actifs stables sont : ", neurones_actifs_stables)
        print("Les neurones inactifs stables sont : ", neurones_inactifs_stables)
        self.L = L_new
        self.U = U_new
        self.neurones_actifs_stables = neurones_actifs_stables
        self.neurones_inactifs_stables = neurones_inactifs_stables
        return L_new, U_new, neurones_actifs_stables, neurones_inactifs_stables
    


    def calcule_bornes_all_algorithms(self,verbose = False, FULL = True):
        
        print("IBP...")
        L_x0_IB, U_x0_IB, neurones_actifs_stables, neurones_inactifs_stables = self.IBP()
        print("Neurones stables trouves apres IBP : ", neurones_actifs_stables+neurones_inactifs_stables)
        print("LP...")
        L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables = self.compute_FULL_U_L(True,
                                                                                        verbose, neurones_actifs_stables, neurones_inactifs_stables)
        if FULL: 
            print("FULL...")
            L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables = self.compute_FULL_U_L(False,
                                                                                            verbose, neurones_actifs_stables, neurones_inactifs_stables)
            print("Neurones stables trouves apres FULL : ", neurones_actifs_stables+neurones_inactifs_stables)
        return L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables
    

    def apply(self, optimization_model: str, parametres_reseau, parametres_optimisation, verbose : bool = False):
        parametres_reseau["L"] = self.L
        parametres_reseau["U"] = self.U
        if optimization_model in optimization_models_gurobi:
            self.apply_gurobi(
                optimization_model, parametres_optimisation, parametres_reseau, verbose = verbose)
        elif optimization_model in optimization_models_mosek:
            self.apply_mosek(
                optimization_model, parametres_optimisation, parametres_reseau, verbose = verbose)
        self.optimization_models.append(optimization_model)

    def apply_mosek(self, optimization_model: str, parametres_optimisation, parametres_reseau, verbose : bool = False):
        ycible = cherche_ycible(self.y0, self.n[self.K])

        coupes_totales = ["RLT_Lan", "zk^2", "betai*betaj","sigmak*zk","betai*zkj"]
        if optimization_model in ["FprG_SDP","FprG_d_SDP"]:
            #coupes_noms = ["RLT_Lan", "zk^2", "betai*betaj","sigmak*zk"]
            coupes_noms = ["zk^2", "betai*betaj","sigmak*zk"]
        elif optimization_model in ["Lan_SDP","Lan_couches_SDP"]:
            #coupes_noms = ["RLT_Lan", "zk^2"]
            coupes_noms = ["zk^2"]
        elif optimization_model in ["Mix_SDP", "Mix_couches_SDP", "Mix_d_SDP", "Mix_d_couches_SDP"]:
            #coupes_noms = ["RLT_Lan", "zk^2", "betai*betaj","betai*zkj"]
            coupes_noms = ["zk^2", "betai*betaj","betai*zkj"]

        dict_coupes_false = {coupe : False for coupe in coupes_totales if coupe not in coupes_noms}
        coupes_combinaisons_model = list(itertools.product([True, False], repeat=len(coupes_noms)))
        dict_coupes_combinaisons_model = [ dict(zip(coupes_noms, combination)) for combination in coupes_combinaisons_model ]
        dict_coupes_combinaisons_model = [{**dic,**dict_coupes_false} for dic in dict_coupes_combinaisons_model]
        
        for coupe in dict_coupes_combinaisons_model:
            parametres_optimisation["coupes"] = coupe
            Sol, opt, status, execution_time, dic_infos = compute_adverse(optimization_model,parametres_reseau, 
            parametres_optimisation, self.x0, self.y0, ycible, self.neurones_actifs_stables, self.neurones_inactifs_stables)
            self.update_resultats(optimization_model, parametres_optimisation, parametres_reseau, ycible, Sol, opt, status, execution_time, dic_infos)


    def apply_gurobi(self, optimization_model: str, parametres_optimisation, parametres_reseau, verbose : bool = False):
        ycible = cherche_ycible(self.y0, self.n[self.K])
        coupes_totales = ["RLT_Lan", "zk^2", "betai*betaj","sigmak*zk","betai*zkj"]
        parametres_optimisation["coupes"] = {coupe_nom : False for coupe_nom in coupes_totales}
        if optimization_model in optimization_models_lineaires:
            parametres_optimisation["relax"] = False
            Sol, opt, status, execution_time, dic_infos = compute_adverse(optimization_model,parametres_reseau, 
            parametres_optimisation, self.x0, self.y0, ycible)
            self.update_resultats(optimization_model, parametres_optimisation, parametres_reseau, ycible, Sol, opt, status, execution_time, dic_infos)
        # parametres_optimisation["relax"] = True
        # Sol, opt, status, execution_time, dic_infos = compute_adverse(optimization_model, parametres_reseau, 
        #     parametres_optimisation, self.x0, self.y0, ycible)
        # self.update_resultats(optimization_model, parametres_optimisation, parametres_reseau, ycible, Sol, opt, status, execution_time, dic_infos)
        
    def solveFprG_SDP_Adv2(self, coupes, titre):
        return MOSEK_FprG_d.solveFprG_SDP_Adv2(self,coupes, titre,verbose = True)
    
    def solveFprG_SDP(self, coupes, titre):
        return MOSEK_FprG.solveFprG_SDP(self,coupes, titre,verbose = True)
    
    def solve_Lan_couches_SDP(self, coupes, titre):
        ycible = cherche_ycible(self.y0, self.n[self.K])
        return MOSEK_Lan_couches.solve_Lan_couches(self,coupes,ycible,titre)
    
    def solve_Lan_SDP(self, coupes, titre):
        ycible = cherche_ycible(self.y0, self.n[self.K])
        return MOSEK_Lan.solve_Lan(self,coupes,ycible,titre)
    
    def solve_Mix_couches_SDP(self, coupes, titre):
        return MOSEK_Mix_couches.solveMix_SDP_par_couches(self,coupes,titre)

    def update_resultats(self, optimization_model: str, parametres_optimisation, parametres_reseau, ycible, Sol, opt, status, execution_time, dic_infos):
        label = -1
        if Sol != []:
            label = self.Res.retourne_label(Sol)
            self.resultats.append({
            'modele_data': self.data_modele,
            'x0': self.x0,
            'x0_id' : self.x0_id,
            'y0' : self.y0,
            'ycible' : ycible,
            'modele_opt': f"{optimization_model}",
            "relax" : parametres_optimisation["relax"],
            'Status' : status,
            'Sol' : Sol,
            'Label' : label  ,
            "Valeur_optimale" : opt,
            'temps_execution': execution_time
        } | parametres_optimisation["coupes"] | dic_infos)
        if Sol != []:
            if self.data_modele == "MNIST":
                print(f"Modèle d'optimisation : {optimization_model};       Status : {status},  Valeur : {opt},     label = {self.Res.retourne_label(Sol)},    temps : {round(execution_time,3)}")
            else :
                print(f"Modèle d'optimisation : {optimization_model};       Status : {status},  Valeur : {opt},      Sol : {Sol}, label = {self.Res.retourne_label(Sol)},    temps : {round(execution_time,3)}")
        else :
            if self.data_modele == "MNIST":
                print(f"Modèle d'optimisation : {optimization_model};       Status : {status},   Valeur : {opt},       temps : {round(execution_time,3)}")
            else :
                print(f"Modèle d'optimisation : {optimization_model};       Status : {status},   Valeur : {opt},      Sol : {Sol},   temps : {round(execution_time,3)}")
        

        


