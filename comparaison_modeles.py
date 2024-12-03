import pandas as pd
import numpy as np
import time
import itertools

import random
import copy
import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from selectionne_x0 import get_x0


dossier_gurobi = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models_gurobi')
sys.path.append(dossier_gurobi)

# GUROBI
from Models_gurobi.GUROBI_outils import parametres_gurobi


from comparaison_modeles_outils import(
    cherche_ycible,
    liste_ycible,
    compute_adverse,
    update_benchmark,
    load_data,
    save_results,
    save_adverses,
    compute_adverses_samples,
    get_subset_from_loader,
    create_folder_benchmark_,
    retourne_weights
)

from calcule_bornes_reseau import (
    compute_FULL_L,
    compute_FULL_U,
    compute_FULL_U_L,
    Interval_Bound_Propagation
)

# RESEAU
from reseau_train import Reseau, architectures_modele





def main():
    data_modele = "MNIST"
    architecture = "2x20"
    n, K = architectures_modele(data_modele,architecture)
    file, data = load_data(data_modele, architecture)
    W, b = retourne_weights(K, n, file)
    print(f"K : {K}, n : {n}")
    
    if data_modele != "MNIST" :
        
        print('W : ', [
            [[round(couche[j][i],2) for i in range(len(couche[0]))] for j in range(len(couche))] 
            for couche in W])
        print('b : ', [[round(couche[i],2) for i in range(len(couche))]  for couche in b])
        
    # ARRONDIS DES PARAMETRES LINEAIRES 
    # W = [[[round(couche[j][i],2) for i in range(len(couche[0]))] for j in range(len(couche))] 
    #     for couche in W]
    # b = [[round(couche[i], 2) for i in range(len(couche))] for couche in b]

    W_reverse = [[ [couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))] for couche in W]
    print(f"Taille W   : { [len(W[i]) for i in range (len(W))]}")
    print(f"Taille W   : { [len(W[i][0]) for i in range (len(W))]}")
    print(f"Taille b   : { [len(b[i]) for i in range (len(b))]}")
    
    # Create network
    Res = Reseau(K, n, W, b)
    if data_modele == "MNIST":
        L, U = Res.retourne_bornes(data_modele="MNIST")
    else : 
        L, U = Res.retourne_bornes()
        print("U  : ", U)
        print("L  : ", L)
    
    parametres_reseau = {"K" : K,
                        "n" : n,
                        "L" : L,
                        "U" : U,
                        "W" : W,
                        "W_reverse" : W_reverse ,
                        "b" : b,
                        }

    # Certification problem hyperparameters
    rho = 0.001
    epsilon_adv = 0.0001
    epsilon = 0.05
    relax = 0
    verbose = True

  
    parametres_optimisation = {"parametres_gurobi" : parametres_gurobi,
                               "rho" : rho,
                               "epsilon" : epsilon,
                               "epsilon_adv" : epsilon_adv,
                               "relax" : relax,
                               "verbose" : verbose}

    optimization_model = "Mix"
    
    #optimization_models_lineaires = ["Glover_Obj_diff","Fischetti_Obj_diff","ReLUconvexe_Adv1"] # Sans Fischetti_dist
    optimization_models_lineaires = ["Glover_Obj_diff","Fischetti_Obj_diff","Fischetti_Obj_dist","ReLUconvexe_Adv1"]
    optimization_models_quadratiques = ["Mix","FprG_quad","Mix_diff_obj_quad","Adv2_ReLU2_lin","F_ReLU1_Adv2",
                                        "ReLU2_Adv2__FprGdiff__","Adv3_ReLU2_lin","F_ReLU1_Adv3","Adv3_ReLU2_lin",
                                        "ReLU3_Adv3","Adv3_ReLU2","F_ReLU1_Adv3","Lan_quad"]
    optimization_models_gurobi = optimization_models_lineaires + optimization_models_quadratiques
    optimization_models_mosek = [ "Mix_SDP","Mix_couches_SDP","Mix_d_SDP","FprG_SDP","Mix_d_couches_SDP","Lan_SDP","Lan_couches_SDP"]
    optimization_models_all = optimization_models_lineaires + optimization_models_mosek

    # CHOIX DES MODELES D'OPTIMISATION A TESTER
    # optimization_models = ["Fischetti_Obj_diff"] +  optimization_models_mosek #+ ["Fischetti_Obj_diff","Mix_diff_obj_quad"]
    optimization_models = ["Fischetti_Obj_diff","Lan_quad","Mix_diff_obj_quad", "Mix_d_SDP","Mix_d_couches_SDP", "Lan_SDP", "Lan_couches_SDP","Mix_SDP", "Mix_couches_SDP"]

    dict_coupes_combinaisons = {}
    # Creation des coupes pour chacun des modeles SDP
    coupes_totales = ["RLT_Lan", "zk^2", "betai*betaj","sigmak*zk","betai*zkj"]
    for model in optimization_models_mosek :
        print("model :", model)
        if model in ["FprG_SDP","FprG_d_SDP"]:
            coupes_noms = ["RLT_Lan", "zk^2", "betai*betaj","sigmak*zk"]
            # coupes_noms = ["zk^2", "betai*betaj","sigmak*zk"]
            coupes_noms = ["zk^2"]

        elif model in ["Lan_SDP","Lan_couches_SDP"]:
            coupes_noms = ["RLT_Lan", "zk^2"]
            #coupes_noms = ["zk^2"]
            coupes_noms = ["zk^2"]

        elif model in ["Mix_SDP", "Mix_couches_SDP", "Mix_d_SDP", "Mix_d_couches_SDP"]:
            coupes_noms = ["RLT_Lan", "zk^2", "betai*betaj","betai*zkj"]
            # coupes_noms = ["zk^2", "betai*betaj","betai*zkj"]
            coupes_noms = ["zk^2"]

        dict_coupes_false = {coupe : False for coupe in coupes_totales if coupe not in coupes_noms}
        coupes_combinaisons_model = list(itertools.product([True, False], repeat=len(coupes_noms)))
        dict_coupes_combinaisons_model = [ dict(zip(coupes_noms, combination)) for combination in coupes_combinaisons_model ]
        dict_coupes_combinaisons_model = [{**dic,**dict_coupes_false} for dic in dict_coupes_combinaisons_model]
        dict_coupes_combinaisons[model] = dict_coupes_combinaisons_model
    
    print("dict coupes combinaisons total : ", dict_coupes_combinaisons)
    
    batch_size = 10
    trainloader = DataLoader(data[0], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(data[1], batch_size=batch_size, shuffle=True)


    # AFFICHAGE DES VALEURS MIN ET MAX DES DONNEES
    for batch in trainloader:
        data, labels = batch
        max_value = data.max()  
        print(f"Valeur maximale du premier batch : {max_value.item()}")
        print(f"Valeur minimale du premier batch : {data.min().item()}")
        break  # On s'arrête après le premier batch
    

    x0_list = get_subset_from_loader(Res, trainloader, n[K], num_elements=1) #+ get_subset_from_loader(Res, testloader, n[K], num_elements=2)

    # Adversarial examples
    # nb_adverses, nb_robustes, adverses, rep = compute_adverses_samples(Res,data_modele,optimization_model,parametres_reseau, parametres_optimisation,x0_list)
    # save_results(rep, data_modele, optimization_model)
    # save_adverses(adverses, data_modele, optimization_model)

    # print("Nombre d'exemples adverses trouvés sur le jeu de données : ", nb_adverses)
    # print("Nombre de samples robustes dans le dataset : ", nb_robustes)
    #x0_list = [(get_x0(), 3)]

    resultats = []
    
    create_folder_benchmark_(data_modele,optimization_models,parametres_reseau,parametres_optimisation,parametres_gurobi,x0_list,resultats)
    for x0_id in range(len(x0_list)):
        x0, y0 = x0_list[x0_id]
        ytrue = y0
        y0 = Res.retourne_label(x0)
        if y0!=ytrue : 
            print(f"Le point x0={x0} n'a pas été sélectionné car mal prédit par le réseau : {y0} au lieu de {ytrue}.")
            continue
        if data_modele == "MNIST" :
            x0 = np.array(x0).reshape(784).tolist()     
        
        ycible = cherche_ycible(ytrue,n[K])
        print(f"ycible = {ycible}")
        # L_x0 = compute_FULL_L(x0,K,n,W_reverse,b,L,U,epsilon)
        # U_x0 = compute_FULL_U(x0,K,n,W_reverse,b,L,U,epsilon)
        # parametres_reseau["L"] = L_x0
        # parametres_reseau["U"] = U_x0
        # print("L_x0 : ", L_x0)
        # print("U_x0 : ", U_x0)
        L_x0_IB, U_x0_IB = Interval_Bound_Propagation(K,n,x0,W_reverse,b,epsilon,verbose=False)
        L_x0, U_x0 = compute_FULL_U_L(x0,K,n,W_reverse,b,L,U,epsilon,verbose=False)
        
        break
        for optimization_model in optimization_models:
            print(f"\n \n \n \n Modèle d'optimisation : {optimization_model}")

            if optimization_model in optimization_models_gurobi : # Si le modele est lineaire ou quadratique (resolution avec gurobi)
                parametres_optimisation["coupes"] = {"zk^2": False, "betai*betaj": False, "sigmak*zk" : False, "RLT_Lan" : False}
                # if optimization_model == "Fischetti_Obj_diff":
                #     print("\n    Discret")
                #     parametres_optimisation["relax"] = 0
                # else : 
                
                #     print("\n    Relaxation")
                #     parametres_optimisation["relax"] = 1

                if optimization_model in optimization_models_lineaires :
                    print("\n    Relaxation")
                    parametres_optimisation["relax"] = 1
                    resultats = update_benchmark(data_modele,optimization_model,parametres_reseau,parametres_optimisation,Res,x0,x0_id,ytrue,ycible,resultats) 
                elif optimization_model in ["Lan_quad"] :
                    ycibles_list = liste_ycible(y0, n[K])
                    for ycible in ycibles_list :
                        print("\n    Discret")
                        print(f"ycible = {ycible}")
                        parametres_optimisation["relax"] = 0
                        resultats = update_benchmark(data_modele,optimization_model,parametres_reseau,parametres_optimisation,Res,x0,x0_id,ytrue,ycible,resultats)
                        print("resultats apres update benchmark : ", resultats)
                print("\n    Discret")
                parametres_optimisation["relax"] = 0
                resultats = update_benchmark(data_modele,optimization_model,parametres_reseau,parametres_optimisation,Res,x0,x0_id,ytrue,ycible,resultats)
                print("resultats apres update benchmark : ", resultats)


            else :   # Si modèle SDP (resolution avec MOSEK)
                for coupes in dict_coupes_combinaisons[optimization_model]:
                    parametres_optimisation["coupes"]   = coupes
                    print(f"\n  \n \n Coupes dictionnaire : {coupes}")
                    if optimization_model in ["Lan_SDP","Lan_couches_SDP"]:
                        ycibles_list = liste_ycible(y0, n[K])
                        for ycible in ycibles_list :
                            resultats = update_benchmark(data_modele,optimization_model,parametres_reseau,parametres_optimisation,Res,x0,x0_id,ytrue,ycible,resultats)
                    else : 
                        resultats = update_benchmark(data_modele,optimization_model,parametres_reseau,parametres_optimisation,Res,x0,x0_id,ytrue,ycible,resultats)
                
            create_folder_benchmark_(data_modele,optimization_models,parametres_reseau,parametres_optimisation,parametres_gurobi,x0_list,resultats, just_change_bench_csv=True,option="")

    create_folder_benchmark_(data_modele,optimization_models,parametres_reseau,parametres_optimisation,parametres_gurobi,x0_list,resultats,just_change_bench_csv=True,option="")

if __name__ == "__main__":
    main()
