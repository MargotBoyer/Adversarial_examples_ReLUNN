import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import time
import random
import abc
import os
import json
import pandas as pd
import pickle
# from reseau_train import Reseau, architectures_modele
from load import load_data, load_file_weights, retourne_weights, cherche_ycible
from reseau_train import Reseau, architectures_modele

from typing import List, Dict

from certification_problem_data import(
     Certification_Problem_Data,
     parametres_gurobi
)

class Certification_Problem(abc.ABC):
    def __init__(self, 
                 data_modele : str, 
                 architecture : str = None, 
                 epsilon : float = 0.1,
                 nb_samples : int = 10):
        
        print("Initialisation du problème de certification...")
        # Load Network
        n, K = architectures_modele(data_modele,architecture)
        file = load_file_weights(data_modele, architecture)
        W, b = retourne_weights(K, n, file)
        self.n, self.K, self.W, self.b = n, K, W, b
        self.W_reverse = [[ [couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))] for couche in W]
        self.Res = Reseau.create_with_file(data_modele, architecture)

        # Load donnees
        data = load_data(data_modele, architecture)
        random_indices = random.sample(range(len(data[0])), nb_samples)
        self.data = Subset(data[0], random_indices)

        self.epsilon = epsilon
        self.data_modele = data_modele
        self.architecture = architecture
        self.resultats = []

        self.nom = f"{self.data_modele}_epsilon={self.epsilon}_taille={len(self.data)}"
        self.create_folder_benchmark_()

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


    def apply(self, verbose : bool = False):
        parametres_reseau = {"K" : self.K,
                    "n" : self.n,
                    "W" : self.W,
                    "W_reverse" : self.W_reverse ,
                    "b" : self.b,
                    }
        parametres_optimisation = dict({"parametres_gurobi" : parametres_gurobi,
                                "rho" : 0.01,
                                "epsilon" : self.epsilon,
                                "epsilon_adv" : 0.01,
                                "verbose" : verbose})
        
        optimizations_models_tester = ["Fischetti_Obj_diff", "Lan_quad", "Mix_diff_obj_quad", 
                                       "Mix_SDP","Mix_couches_SDP","Mix_d_SDP",
                                       "FprG_SDP","Mix_d_couches_SDP","Lan_SDP",
                                       "Lan_couches_SDP"]
        for ind_x0 in range(len(self.data)):
            x0, y0 = self.data[ind_x0]
            x0 = x0.view(-1)
            print(f"Shape x0 : {x0.shape}")
            print(f"Application pour x0 : {x0} et y0 : {y0}")
            cert = Certification_Problem_Data(self.data_modele, self.architecture, 
                                              x0, y0.item(), ind_x0, self.epsilon)
            #cert.IBP()
            cert.calcule_bornes_all_algorithms(verbose = verbose, FULL = True)
            for optimization_model in optimizations_models_tester: 
                cert.apply(optimization_model, parametres_reseau, parametres_optimisation, verbose = True)
                self.update_folder_benchmark_(parametres_reseau, parametres_optimisation, parametres_gurobi, cert, just_change_bench_csv = True, option = "")
                #print(f"Resultats pour l'instant pour le probleme de certification sur la donnee numero {ind_x0} : ", cert.resultats)
            self.resultats.extend(cert.resultats)
            #print("Résultats : ", self.resultats)
    
    
    def test(self):
        coupes_liste = [{"RLT_Lan" : False, 
                  "zk^2" : False,
                  "betai*betaj" : False,
                  "sigmak*zk" : False,
                  "betai*zkj" : False},

                  {"RLT_Lan" : False, 
                  "zk^2" : True,
                  "betai*betaj" : False,
                  "sigmak*zk" : False,
                  "betai*zkj" : False},

                  {"RLT_Lan" : True, 
                  "zk^2" : False,
                  "betai*betaj" : False,
                  "sigmak*zk" : False,
                  "betai*zkj" : False}
                  ]
        
        coupes_liste = [
                  {"RLT_Lan" : False, 
                  "zk^2" : True,
                  "betai*betaj" : False,
                  "sigmak*zk" : False,
                  "betai*zkj" : False}
                  ]
        
        print("n ", self.n)
        
        for ind_x0 in range(len(self.data)):
            x0, y0 = self.data[ind_x0]
            x0 = x0.view(-1)
            print(f"Shape x0 : {x0.shape}")
            print(f"Application pour x0 : {x0} et y0 : {y0}")
            time.sleep(2)
            cert = Certification_Problem_Data(self.data_modele, self.architecture, 
                                              x0, y0.item(), ind_x0, self.epsilon)
            cert.calcule_bornes_all_algorithms()
            folder_dir = f"datasets\{self.data_modele}\Benchmark\{self.nom}"
            if not os.path.exists(folder_dir):
                print("Creation du dossier...")
                os.makedirs(folder_dir)

            # cert.solve_ReLU_convexe(True,self.nom)
            # cert.solve_Mix_d(True, self.nom)
            # cert.solve_Fischetti_diff(True,self.nom)
            # cert.solve_FprG(True, self.nom)
            # cert.solve_Lan_quad(True, self.nom)

            for coupes in coupes_liste:
                print("Coupes : ", coupes)
                # cert.solve_Lan_couches_SDP(coupes,self.nom)
                # cert.solve_Lan_SDP(coupes,self.nom)
                # cert.solve_Mix_couches_SDP(coupes, self.nom)
                # cert.solve_Mix_d_couches_SDP(coupes,self.nom)
                # print("Solve Mix_d")
                # cert.solve_Mix_d_SDP(coupes, self.nom)
                # cert.solve_Mix_SDP(coupes, self.nom)
                # cert.solveFprG_SDP(coupes, self.nom)
                # cert.solveFprG_SDP_Adv2(coupes, self.nom)
            

    def create_folder_benchmark_(self,
        just_change_bench_csv : bool = False):
        """ Création du dossier de résultats """
        folder_dir = f"datasets\{self.data_modele}\Benchmark\{self.nom}"
        if not os.path.exists(folder_dir) and not just_change_bench_csv:
            print("Creation du dossier...")
            os.makedirs(folder_dir)

        os.makedirs(folder_dir, exist_ok=True)


    def update_folder_benchmark_(self,
        parametres_reseau: Dict,
        parametres_optimization: Dict,
        parametres_gurobi: Dict,
        cert : Certification_Problem_Data,
        just_change_bench_csv : bool = False, 
        option : str = None):
        """ Enregistre les résultats du run dans un dossier """
    
        # Création du chemin de dossier
        folder_dir = f"datasets\{self.data_modele}\Benchmark\{self.nom}"

        resultats_file_name = f"{self.data_modele}_epsilon={self.epsilon}_neurones={sum(self.n[1:self.K+1])}_taille={len(self.data)}_{option}_benchmark.csv"
        resultats_file_path = os.path.join(folder_dir, resultats_file_name)
        
        # Supprimer le fichier de résultats s'il existe déjà
        if just_change_bench_csv and os.path.exists(resultats_file_path):
            print("Remove active")
            #print("resultats file path : ", resultats_file_path)
            os.remove(resultats_file_path)
            #print(f"Fichier existant supprimé : {resultats_file_path}")
        
        # Sauvegarde des données
        list_file_path = os.path.join(folder_dir, 'x0_liste.obj')

        with open(list_file_path, 'wb') as list_file:
            pickle.dump(self.data, list_file)

        dict_file_path_reseau = os.path.join(folder_dir, 'Parametres_reseau.json')
        with open(dict_file_path_reseau, 'w') as dict_file:
            json.dump({"Data_modele": data_modele,
                    "Optimization_models": cert.optimization_models} |    
                    parametres_reseau, dict_file, indent=4)
            #print(f"Dictionnaire des paramètres réseau enregistré dans : {dict_file_path_reseau}")

        dict_file_path_optimization = os.path.join(folder_dir, 'Parametres_optimization.json')
        with open(dict_file_path_optimization, 'w') as dict_file:
            json.dump(parametres_optimization, dict_file, indent=4)
            #print(f"Dictionnaire des paramètres optimisation enregistré dans : {dict_file_path_optimization}")

        dict_file_path_gurobi = os.path.join(folder_dir, 'Parametres_gurobi.json')
        with open(dict_file_path_gurobi, 'w') as dict_file:
            json.dump(parametres_gurobi, dict_file, indent=4)
            #print(f"Dictionnaire des paramètres de gurobi enregistré dans : {dict_file_path_gurobi}")

        # Enregistrement du fichier CSV de résultats
        if cert.resultats != []:
            resultats_df = pd.DataFrame(self.resultats)
            resultats_df.to_csv(resultats_file_path, index=False)
            print(f"Résultats enregistrés dans : {resultats_file_path}")






if __name__ == "__main__":

    data_modele = "BLOB"
    architecture = None
    epsilon = 10
    Certification_Problem_ = Certification_Problem(data_modele, architecture, epsilon, nb_samples=1)
    print("Data : ", Certification_Problem_.data)
    Certification_Problem_.test()
    
    # x0, y0 = Certification_Problem_MNIST.data[0][0][0], Certification_Problem_MNIST.data[0][0][1]
    # print("x0 : ", x0)
    # x0 = np.array(x0).reshape(784).tolist()  
    #L, U, neurones_actifs_stables, neurones_inactifs_stables = Certification_Problem_MNIST.IBP(torch.tensor(x0))
    #L, U, neurones_actifs_stables, neurones_inactifs_stables = Certification_Problem_MNIST.compute_FULL_U_L(x0, L, U, verbose = True, neurones_actifs_stables = neurones_actifs_stables, neurones_inactifs_stables = neurones_inactifs_stables)
    #L, U, neurones_actifs_stables, neurones_actifs_stables = Certification_Problem_MNIST.calcule_bornes_all_algorithms(torch.tensor(x0), verbose = False)





