import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import time
import datetime
import random
import abc
import os
import json
import pandas as pd
import pickle
# from reseau_train import Reseau, architectures_modele
from load import load_data, load_file_weights, retourne_weights, cherche_ycible
from reseau_train import Reseau, architectures_modele, device

from typing import List, Dict

from certification_problem_data import(
     Certification_Problem_Data,
     parametres_gurobi,
     optimization_models_mosek
)
from comparaison_modeles_outils import (
    remove_folder_benchmark
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
        self.Res = Reseau.create_with_file(data_modele, architecture).to(device)

        # Load donnees
        data = load_data(data_modele, architecture)
        random_indices = random.sample(range(len(data[0])), nb_samples)
        self.data = Subset(data[0], random_indices)

        self.epsilon = epsilon
        self.data_modele = data_modele
        self.architecture = architecture
        self.resultats = []

        self.nom = f"{self.data_modele}_epsilon={self.epsilon}_taille={len(self.data)}_" + "{:%d_%b_%y_%Hh_%M}".format(datetime.datetime.now())
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


    def apply(self, verbose : bool = False, coupes = None):
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
        
        # optimizations_models_tester = ["Fischetti_Obj_diff", "Lan_quad", "Mix_diff_obj_quad", 
        #                                "Mix_SDP","Mix_couches_SDP","Mix_d_SDP",
        #                                "FprG_SDP","Mix_d_couches_SDP","Lan_SDP",
        #                                "Lan_couches_SDP"]

        optimizations_models_tester =["Fischetti_Obj_diff", "Mix_d_SDP", "Mix_d_couches_SDP","Lan_SDP", "Lan_couches_SDP",  "Mix_SDP","Mix_couches_SDP"]
        optimizations_models_tester =["Fischetti_Obj_diff","Mix_SDP"]
        
        print("n : ", self.n)
        for i in range(len(self.W)):
            print(f"\n \n \n Couche {i}")
            for j in range(len(self.W[i])):
                print("W : ", [round(self.W[i][j][k]) for k in range(len(self.W[i][j]))])
            print("b : ", [round(self.b[i][j]) for j in range(len(self.b[i]))])
        
        for ind_x0 in range(len(self.data)):
            x0, y0 = self.data[ind_x0]
            x0 = x0.view(-1)
            print(f"Shape x0 : {x0.shape}")
            print(f"Application pour x0 : {x0} et y0 : {y0}")
            if isinstance(y0, torch.Tensor):
                y0 = y0.item()
            cert = Certification_Problem_Data(self.data_modele, self.architecture, 
                                              x0, y0, ind_x0, self.epsilon)
            #cert.IBP()
            cert.calcule_bornes_all_algorithms(verbose = verbose, FULL = False)
            print("U : ", cert.U)
            print("L : ", cert.L)
            print("neurones_actifs_stables : ", cert.neurones_actifs_stables)
            print("neurones_inactifs_stables : ", cert.neurones_inactifs_stables)
            for k in range(self.K):
                for j in range(self.n[k]):
                    if (k,j) not in cert.neurones_actifs_stables + cert.neurones_inactifs_stables:
                        print(f"Neurone ({k},{j}) : "
                                f"U = {cert.U[k][j]} et L = {cert.L[k][j]}")    
           
            for optimization_model in optimizations_models_tester: 
                model_dir = f"datasets/{self.data_modele}/Benchmark/{self.nom}/{optimization_model}"
                if (not os.path.exists(model_dir)) and (optimization_model in optimization_models_mosek):
                    print(f"Création du dossier : {model_dir}")
                    os.makedirs(model_dir)
                cert.apply(optimization_model, parametres_reseau, parametres_optimisation, titre = self.nom, 
                           derniere_couche_lineaire= True, coupes = coupes, verbose = verbose)
                self.update_folder_benchmark_(parametres_reseau, parametres_optimisation, parametres_gurobi, cert, just_change_bench_csv = True, option = "")
                #print(f"Resultats pour l'instant pour le probleme de certification sur la donnee numero {ind_x0} : ", cert.resultats)
            self.resultats.extend(cert.resultats)
            #print("Résultats : ", self.resultats)
    
    
    def test(self):        
        coupes_liste = [
                  {"RLTLan" : True, 
                  "zk2" : True,
                  "betaibetaj" : False,
                  "sigmakzk" : False,
                  "betaizkj" : False,
                  "bornes_betaz" : False}
                  ]
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
                                "verbose" : True})
        
        print("n ", self.n)
        #optimizations_models_tester  = ["Lan_SDP", "Lan_couches_SDP", "Mix_d_SDP", "Mix_d_couches_SDP", "Mix_SDP", "Mix_couches_SDP", "FprG_SDP"]
        optimizations_models_tester  = ["Mix_SDP"]

        for ind_x0 in range(len(self.data)):
            x0, y0 = self.data[ind_x0]
            x0 = x0.view(-1)
            print(f"Shape x0 : {x0.shape}")
            print(f"Application pour x0 : {x0} et y0 : {y0}")
            ycible = cherche_ycible(y0, self.n[self.K])
            #time.sleep(2)
            if isinstance(y0, torch.Tensor):
                y0 = y0.item()
            cert = Certification_Problem_Data(self.data_modele, self.architecture, 
                                              x0, y0, ind_x0, self.epsilon)
            #cert.calcule_bornes_all_algorithms()
            cert.IBP()
            print("U : ", cert.U)
            print("L : ", cert.L)

            cert.solve("Fischetti_Obj_diff", self.nom, relax = False)

            folder_dir = f"datasets/{self.data_modele}/Benchmark/{self.nom}"
            if not os.path.exists(folder_dir):
                print("Creation du dossier...")
                os.makedirs(folder_dir)


            for optimization_model in optimizations_models_tester: 
                model_dir = os.path.join(folder_dir, optimization_model)
                if (not os.path.exists(model_dir)) and (optimization_model in optimization_models_mosek):
                    print(f"Création du dossier : {model_dir}")
                    os.makedirs(model_dir)

                print("\n AVEC la derniere couche dans la matrice variable :",)
                Sol, opt, status, execution_time, dic_infos = cert.solve(optimization_model, self.nom, coupes = coupes_liste[0], derniere_couche_lineaire=True, verbose = False)
                print("Opt : ", opt)
                print("\n SANS la derniere couche dans la matrice variable :",)
                Sol, opt, status, execution_time, dic_infos = cert.solve(optimization_model, self.nom, coupes = coupes_liste[0], derniere_couche_lineaire=False, verbose = False)
                print("Opt : ", opt)
                # for coupes in coupes_liste:
                #     print("Coupes : ", coupes)
                #     Sol, opt, status, execution_time, dic_infos = cert.solve(optimization_model, self.nom, coupes = coupes)
                    # cert.update_resultats(optimization_model, parametres_optimisation, 
                    #                       parametres_reseau, ycible, Sol, opt, status, execution_time, dic_infos)

            


    def create_folder_benchmark_(self,
        just_change_bench_csv : bool = False):
        """ Création du dossier de résultats """
        folder_dir = os.path.join(os.getcwd(), f"datasets/{self.data_modele}/Benchmark/{self.nom}")
        if not os.path.exists(folder_dir) and not just_change_bench_csv:
            print("Creation du dossier...")
            os.makedirs(folder_dir)

        #os.makedirs(folder_dir, exist_ok=True)


    def update_folder_benchmark_(self,
        parametres_reseau: Dict,
        parametres_optimization: Dict,
        parametres_gurobi: Dict,
        cert : Certification_Problem_Data,
        just_change_bench_csv : bool = False, 
        option : str = None):
        """ Enregistre les résultats du run dans un dossier """
    
        # Création du chemin de dossier
        folder_dir = os.path.join(os.getcwd(), f"datasets/{self.data_modele}/Benchmark/{self.nom}")

        if option is not None :
            resultats_file_name = f"{self.data_modele}_epsilon={self.epsilon}_neurones={sum(self.n[1:self.K+1])}_taille={len(self.data)}_{option}_benchmark.csv"
        else : 
            resultats_file_name = f"{self.data_modele}_epsilon={self.epsilon}_neurones={sum(self.n[1:self.K+1])}_taille={len(self.data)}_benchmark.csv"
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

        # print("cert resultats : ", cert.resultats)
        # Enregistrement du fichier CSV de résultats
        if cert.resultats != []:
            print("Les resultats ne sont pas vides")
            resultats_df = pd.DataFrame(cert.resultats)
            resultats_df.to_csv(resultats_file_path, index=False)
            print("Resultats df : ", resultats_df)
            print(f"Résultats enregistrés dans : {resultats_file_path}")






if __name__ == "__main__":

    data_modele = "BLOB"
    #remove_folder_benchmark(data_modele)

    architecture = None
    epsilon = 5
    Certification_Problem_ = Certification_Problem(data_modele, architecture, epsilon, nb_samples=1)
    print("Data : ", Certification_Problem_.data)
    Certification_Problem_.apply(verbose = False, coupes = ["zk2"])
    
    # x0, y0 = Certification_Problem_MNIST.data[0][0][0], Certification_Problem_MNIST.data[0][0][1]
    # print("x0 : ", x0)
    # x0 = np.array(x0).reshape(784).tolist()  
    #L, U, neurones_actifs_stables, neurones_inactifs_stables = Certification_Problem_MNIST.IBP(torch.tensor(x0))
    #L, U, neurones_actifs_stables, neurones_inactifs_stables = Certification_Problem_MNIST.compute_FULL_U_L(x0, L, U, verbose = True, neurones_actifs_stables = neurones_actifs_stables, neurones_inactifs_stables = neurones_inactifs_stables)
    #L, U, neurones_actifs_stables, neurones_actifs_stables = Certification_Problem_MNIST.calcule_bornes_all_algorithms(torch.tensor(x0), verbose = False)





