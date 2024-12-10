import numpy as np
import pandas as pd
import torch
import random

import os
import sys
import json
import shutil
import pickle
from typing import Dict, List, Tuple

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Construire le chemin absolu vers le dossier des modèles SDP (afin de pouvoir faire des import d'import)
dossier_mosek = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models_MOSEK')
# Construire le chemin absolu vers le dossier des modèles résolus avec gurobi
dossier_gurobi = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models_gurobi')

sys.path.append(dossier_mosek)
sys.path.append(dossier_gurobi)

# GUROBI
#from Models_gurobi.GUROBI_outils import parametres_gurobi

from Models_gurobi.Model_ReLU3_Adv1__Mix__ import solveMix
from Models_gurobi.Model_ReLU3_Adv2__Mixdiff__ import solveMix_diff_obj_quad
from Models_gurobi.Model_ReLU1_Adv1_Fischetti_diff import solveFischetti_Objdiff
from Models_gurobi.Model_ReLU1_dist__Fischetti__ import solveFischetti_Objdist
from Models_gurobi.Model_ReLU2lin_Adv1__G__ import solveGlover_diff
from Models_gurobi.Model_ReLU2_Adv1__FprG__ import solveFprG_quad
from Models_gurobi.Model_ReLU2lin_Adv2 import solveG_RelU2_lin_Adv2
from Models_gurobi.Model_ReLU1_Adv2 import solve_F_ReLU1_Adv2
from Models_gurobi.Model_ReLU2_Adv2 import solve_FprG_ReLU2_Adv2
from Models_gurobi.Model_ReLUconvexe_Adv1 import solve_ReLUconvexe_Adv1
from Models_gurobi.Model_ReLU1_Adv3 import solve_F_ReLU1_Adv3
from Models_gurobi.Model_ReLU2lin_Adv3 import solveG_RelU2_lin_Adv3
from Models_gurobi.Model_ReLU3_Adv3 import solve_ReLU3_Adv3
from Models_gurobi.Model_ReLU2_Adv3 import solve_FprG_ReLU2_Adv3
from Models_gurobi.Model_ReLU3_Advcible_Lan import solve_Lan_quad

# MOSEK
from Models_MOSEK.Model_FprG import solveFprG_SDP
from Models_MOSEK.Model_Mix import solveMix_SDP
from Models_MOSEK.Model_Mix_couches import solveMix_SDP_par_couches
from Models_MOSEK.Model_Mix_d import solveMix_SDP_objbetas
from Models_MOSEK.Model_Mix_d_couches import solve_Mix_SDP_objbetas_couches
from Models_MOSEK.Model_Lan import solve_Lan
from Models_MOSEK.Model_Lan_couches import solve_Lan_couches


def cherche_ycible(ytrue : int,
                   nK : int):
    """ Renvoie une classe cible aléatoire """
    possible_values = [i for i in range(nK) if i != ytrue]
    return random.choice(possible_values)

def liste_ycible(ytrue : int,
                   nK : int):
    """ Renvoie une liste avec toutes les classes cibles adversariales possibles """
    return [i for i in range(nK) if i != ytrue]



def retourne_weights(K : int, 
                     n : List[int], 
                     file : str):
    parametres = torch.load(file)
    W = []
    b = []
    print("Retourne weights activé")
    for k in range(K):
        weight_k = f"layers.{2 * k}.weight"
        bias_k = f"layers.{2 * k}.bias"
        W.append(parametres[weight_k].tolist())
        b.append(parametres[bias_k].tolist())
    return W, b

def compute_adverse(
        optimization_model : str, 
        parametres_reseau : Dict, 
        parametres_optimisation : Dict, 
        x0 : List[float], 
        ytrue : int, 
        ycible : int):
    """ Calcul du problème de certification sur la donnée x0 de label ytrue avec pour modèle optimization_modele"""
    if optimization_model == "Fischetti_Obj_dist":
        if parametres_optimisation["verbose"]:
            print(" \n   Fischetti_Obj_dist:  ")
        Sol, opt, status, time, dic_infos = solveFischetti_Objdist(parametres_reseau["K"], parametres_reseau['n'], x0, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["rho"], parametres_optimisation["epsilon"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
    elif optimization_model == "Fischetti_Obj_diff":
        if parametres_optimisation["verbose"]:
            print(" \n   Fischetti_Obj-diff  : ")
        Sol, opt, status, time, dic_infos = solveFischetti_Objdiff(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["rho"], parametres_optimisation["epsilon"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "F_ReLU1_Adv2":
        if parametres_optimisation["verbose"]:
            print(" \n   F_ReLU1_Adv2  : ")
        Sol, opt, status, time, dic_infos = solve_F_ReLU1_Adv2(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["rho"], parametres_optimisation["epsilon"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
    elif optimization_model == "F_ReLU1_Adv3":
        if parametres_optimisation["verbose"]:
            print(" \n   F_ReLU1_Adv3  : ")
        Sol, opt, status, time, dic_infos = solve_F_ReLU1_Adv3(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["rho"], parametres_optimisation["epsilon"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "ReLUconvexe_Adv1":
        if parametres_optimisation["verbose"]:
            print(" \n   ReLUconvexe_Adv1  : ")
        Sol, opt, status, time, dic_infos = solve_ReLUconvexe_Adv1(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["rho"], parametres_optimisation["epsilon"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "Mix":
        if parametres_optimisation["verbose"]:
            print(" \n  Mix : ")
        Sol, opt, status, time, dic_infos = solveMix(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "ReLU3_Adv3":
        if parametres_optimisation["verbose"]:
            print(" \n  ReLU3_Adv3 : ")
        Sol, opt, status, time, dic_infos = solve_ReLU3_Adv3(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "Glover_Obj_diff":
        if parametres_optimisation["verbose"]:
            print(" \n    Glover_Obj-diff : ")
        Sol, opt, status, time, dic_infos = solveGlover_diff(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "Adv2_ReLU2_lin":
        if parametres_optimisation["verbose"]:
            print(" \n    Adv2_ReLU2_lin : ")
        Sol, opt, status, time, dic_infos = solveG_RelU2_lin_Adv2(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "Adv3_ReLU2_lin":
        if parametres_optimisation["verbose"]:
            print(" \n    Adv3_ReLU2_lin : ")
        Sol, opt, status, time, dic_infos = solveG_RelU2_lin_Adv3(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
    
    elif optimization_model == "Adv3_ReLU2":
        if parametres_optimisation["verbose"]:
            print(" \n    Adv3_ReLU2 : ")
        Sol, opt, status, time, dic_infos = solve_FprG_ReLU2_Adv3(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], parametres_optimisation["relax"],
                                                        parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])

    elif optimization_model == "Mix_SDP":
        if parametres_optimisation["verbose"]:
            print(" \n    Mix_SDP : ")
        Sol, opt, status, time, dic_infos = solveMix_SDP(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["rho"],
                                                        parametres_optimisation["coupes"], parametres_optimisation["verbose"])
    elif optimization_model == "FprG_SDP" :
        if parametres_optimisation["verbose"]:
            print(" \n    FprG_SDP : ")
        Sol, opt, status, time, dic_infos = solveFprG_SDP(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["rho"], 
                                                        parametres_optimisation["coupes"], parametres_optimisation["verbose"])
    elif optimization_model == "FprG_quad" :
        if parametres_optimisation["verbose"]:
            print(" \n    FprG_quad : ")
        Sol, opt, status, time, dic_infos = solveFprG_quad(parametres_reseau["K"], parametres_reseau['n'],x0,ytrue,
                                                parametres_reseau["U"], parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"],
                                                parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], 
                                                        parametres_optimisation["relax"], parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "ReLU2_Adv2__FprGdiff__" :
        if parametres_optimisation["verbose"]:
            print(" \n    ReLU2_Adv2__FprGdiff__ : ")
        Sol, opt, status, time, dic_infos = solve_FprG_ReLU2_Adv2(parametres_reseau["K"], parametres_reseau['n'],x0,ytrue,
                                                parametres_reseau["U"], parametres_reseau["L"],  parametres_reseau["W_reverse"],  parametres_reseau["b"],
                                                parametres_optimisation["epsilon"], parametres_optimisation["epsilon_adv"], 
                                                        parametres_optimisation["relax"], parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    elif optimization_model == "Mix_couches_SDP":
        if parametres_optimisation["verbose"]:
            print(" \n    Mix_couches_SDP : ")
        Sol, opt, status, time, dic_infos = solveMix_SDP_par_couches(parametres_reseau["K"], parametres_reseau['n'], x0, ytrue, ycible, parametres_reseau["U"], 
                                                        parametres_reseau["L"],  parametres_reseau["W"],  parametres_reseau["b"], 
                                                        parametres_optimisation["epsilon"], parametres_optimisation["rho"],
                                                        parametres_optimisation["coupes"], parametres_optimisation["verbose"])
    elif optimization_model == "Mix_d_SDP":
        if parametres_optimisation["verbose"]:
            print(" \n    Mix_d_SDP : ")
        Sol, opt, status, time, dic_infos = solveMix_SDP_objbetas(parametres_reseau["K"], parametres_reseau["n"], x0, ytrue, 
                                                       ycible, parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"], parametres_optimisation["rho"], 
                                                       parametres_optimisation["coupes"], parametres_optimisation["verbose"])
    elif optimization_model == "Mix_d_couches_SDP":
        if parametres_optimisation["verbose"]:
            print(" \n    Mix_d_couches_SDP : ")
        Sol, opt, status, time, dic_infos = solve_Mix_SDP_objbetas_couches(parametres_reseau["K"], parametres_reseau["n"], x0, ytrue, 
                                                       ycible, parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"], parametres_optimisation["rho"], 
                                                       parametres_optimisation["coupes"], parametres_optimisation["verbose"])
    elif optimization_model == "Mix_diff_obj_quad" :
        if parametres_optimisation["verbose"]:
            print(" \n    Mix_diff_obj_quad : ")
        Sol, opt, status, time, dic_infos = solveMix_diff_obj_quad(parametres_reseau["K"], parametres_reseau["n"],x0,ytrue,ycible,
                                                          parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W_reverse"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"],
                                                       parametres_optimisation["epsilon_adv"], 
                                                       parametres_optimisation["relax"], parametres_optimisation["parametres_gurobi"],parametres_optimisation["verbose"])
    elif optimization_model == "Lan_SDP" :
        if parametres_optimisation["verbose"]:
            print(" \n    Lan_SDP : ")
        Sol, opt, status, time, dic_infos = solve_Lan(parametres_reseau["K"], parametres_reseau["n"],x0,ytrue,ycible,
                                                          parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"],
                                                       parametres_optimisation["rho"], parametres_optimisation["coupes"], 
                                                      parametres_optimisation["verbose"])
    elif optimization_model == "Lan_couches_SDP" :
        if parametres_optimisation["verbose"]:
            print(" \n    Lan_couches_SDP : ")
        Sol, opt, status, time, dic_infos = solve_Lan_couches(parametres_reseau["K"], parametres_reseau["n"],x0,ytrue,ycible,
                                                          parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"],
                                                       parametres_optimisation["rho"], parametres_optimisation["coupes"], 
                                                      parametres_optimisation["verbose"])
        
    elif optimization_model == "Lan_quad" :
        if parametres_optimisation["verbose"]:
            print(" \n    Lan_quad : ")
        Sol, opt, status, time, dic_infos = solve_Lan_quad(parametres_reseau["K"], parametres_reseau["n"],x0,ytrue,ycible,
                                                          parametres_reseau["U"], parametres_reseau["L"], parametres_reseau["W_reverse"], 
                                                       parametres_reseau["b"], parametres_optimisation["epsilon"],
                                                      parametres_optimisation["parametres_gurobi"], parametres_optimisation["verbose"])
        
    else:
        print(f"Le nom du modèle ne correspond pas aux modèles inscrits : {optimization_model}")
        return None, None, None, None, None
    return Sol, opt, status, time, dic_infos


def update_benchmark(
        data_modele : str,
        optimization_model : str,
        parametres_reseau : Dict,
        parametres_optimisation : Dict,
        Res : torch.nn.Module,
        x0 : List[float],
        x0_id : int,
        ytrue : int,
        ycible : int,
        resultats : List):
    
    """ Calcul du problème de certification sur la donnée x0, de classe ytrue, avec pour modele d'optimization optimization_model, 
    et enregistrement et log des résultats """
    Sol, opt, status, execution_time, dic_infos = compute_adverse(optimization_model, parametres_reseau, parametres_optimisation, x0, ytrue, ycible)
    label = -1
    if Sol != []:
        label = Res.retourne_label(Sol)
    resultats.append({
        'modele_data': data_modele,
        'x0': x0,
        'x0_id' : x0_id,
        'y0' : ytrue,
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
        if data_modele == "MNIST":
            print(f"Modèle d'optimisation : {optimization_model};       Status : {status},  Valeur : {opt},     label = {Res.retourne_label(Sol)},    temps : {round(execution_time,3)}")
        else :
            print(f"Modèle d'optimisation : {optimization_model};       Status : {status},  Valeur : {opt},      Sol : {Sol}, label = {Res.retourne_label(Sol)},    temps : {round(execution_time,3)}")
    else :
        if data_modele == "MNIST":
            print(f"Modèle d'optimisation : {optimization_model};       Status : {status},   Valeur : {opt},       temps : {round(execution_time,3)}")
        else :
            print(f"Modèle d'optimisation : {optimization_model};       Status : {status},   Valeur : {opt},      Sol : {Sol},   temps : {round(execution_time,3)}")
    return resultats


def load_data(data_modele : str, architecture : str = None):
    if data_modele == "MOON":
        data = torch.load('datasets/MOON/MOON_dataset.pt')

    elif data_modele == "BLOB":
        data = torch.load('datasets/BLOB/BLOB_dataset.pt')

    elif data_modele == "MULTIPLE_BLOB":
        data = torch.load('datasets/MULTIPLE_BLOB/MULTIPLE_BLOB_dataset.pt')

    elif data_modele == "CIRCLE" :
        data = torch.load('datasets/CIRCLE/CIRCLE_dataset.pt')

    elif data_modele == "MULTIPLE_CIRCLES" :
        print("on selectionne des donnees de type multiple circles")
        data = torch.load('datasets/MULTIPLE_CIRCLES/MULTIPLE_CIRCLES_noise=0.15_n_circles=2_dataset.pt')

    elif data_modele == "MULTIPLE_DIM_GAUSSIANS":
        data = torch.load('datasets/MULTIPLE_DIM_GAUSSIANS/MULTIPLE_DIM_GAUSSIANS_dataset.pt')

    elif data_modele == "MNIST" :
        # transform = transforms.Compose([transforms.ToTensor())])
        # train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        # test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        # data = [train_dataset, test_dataset]
        data = torch.load('datasets/MNIST/critical_points_dataset_seuil=0.5.pt')
    print("Data loaded !  \n ", data[0])
    return data


def load_file_weights(data_modele : str, architecture : str = None):
    if data_modele == "MOON":
        file = 'datasets/MOON/Res_Moon_weights.pth'

    elif data_modele == "BLOB":
        file = 'datasets/BLOB/Res_BLOB_weights.pth'

    elif data_modele == "MULTIPLE_BLOB":
        file = 'datasets/MULTIPLE_BLOB/Res_MULTIPLE_BLOB_weights.pth'

    elif data_modele == "CIRCLE" :
        file = 'datasets/CIRCLE/Res_CIRCLE_weights.pth'

    elif data_modele == "MULTIPLE_CIRCLES" :
        print("on selectionne des donnees de type multiple circles")
        file = 'datasets/MULTIPLE_CIRCLES/Res_MULTIPLE_CIRCLES_weights.pth'

    elif data_modele == "MULTIPLE_DIM_GAUSSIANS":
        file = 'datasets/MULTIPLE_DIM_GAUSSIANS/Res_MULTIPLE_DIM_GAUSSIANS_weights.pth'

    elif data_modele == "MNIST" :
        file = 'datasets/MNIST/Res_MNIST_6x100_weights.pth'
        if architecture is not None : 
            file = f'datasets/MNIST/Res_MNIST_{architecture}_weights.pth'
    return file




def save_results(
        rep : pd.DataFrame, 
        data_modele : str, 
        optimization_model : str):
    rep.to_csv(f"datasets/{data_modele}_{optimization_model}_benchmark.csv")


def save_adverses(adverses : List[tuple[List[float],int]], data_modele : str, optimization_model : str):
    with open(f'datasets/{data_modele}_{optimization_model}_adverses.pkl', 'wb') as f:
        pickle.dump(adverses, f)


def compute_adverses_samples(
        Res : torch.nn.Module,
        data_modele : str,
        optimization_model : str,
        parametres_reseau : Dict, 
        parametres_optimisation : Dict,
        x0_list : List[Tuple[List[float],int]]):
    """ Calcul des exemples adverses à partir des données de x0_list """
    adverses = []
    nb_adverses = 0
    nb_robustes = 0

    cols = ['K', 'n', 'W', 'b', 'x0', 'xo_ID', 'ytrue', 'ycible', 'epsilon', 'epsilon_adv', 'rho',
            f'{optimization_model}-Status', f'{optimization_model}-Solution', f'{optimization_model}-Temps']

    rep = pd.DataFrame(columns=cols)
    
    for x0_id in range(len(x0_list)):
        x0, y0 = x0_list[x0_id]
        ytrue = y0
        ycible = ytrue ^ 1

        Sol, status, opt, time = compute_adverse(optimization_model, parametres_reseau, parametres_optimisation, x0, ytrue, ycible)

        rep.loc[len(rep)] = [parametres_reseau["K"], parametres_reseau["n"], parametres_reseau["W"]
                             , parametres_reseau["b"], x0, x0_id, ytrue, ycible, parametres_optimisation["epsilon"]
                             , parametres_optimisation["epsilon_adv"], parametres_optimisation["rho"], status, Sol, float(time)]
        
        print("status : ", status)
        if status == 1:
            classe_adv = np.argmax(Res(Sol))
            if classe_adv == ytrue:
                print("L'exemple trouve n'est pas un exemple adverse !!")
                continue
            adverses.append((Sol, classe_adv, x0, ytrue))
            nb_adverses += 1
        else:
            nb_robustes += 1

    return nb_adverses, nb_robustes, adverses, rep


def get_subset_from_loader(
        Res : torch.nn.Module, 
        loader : DataLoader, 
        n_classes : int, 
        num_elements : int = 50):
    """ Renvoie la liste des données sur lesquelles tester le probleme de certification """
    x0_list = []
    classes_found = {classe: False for classe in range(n_classes)}
    for batch in loader : 
        inputs, labels = batch
        for i in range(inputs.shape[0]):
            x0 = inputs[i].tolist()
            ytrue = labels[i].item()

            ypred = Res.retourne_label(x0)

            if ytrue != ypred:
                print(f"On ne selectionne pas cette donnée de label {ytrue} car elle est mal prédite par le réseau.")
                continue
            #classes_found[ytrue] = True
            if num_elements >= n_classes and len(x0_list) < num_elements:
                if all(classes_found.values()):
                    x0_list.append((x0,ytrue))
                    classes_found[ytrue] = True
                elif not classes_found[ytrue]:
                    x0_list.append((x0,ytrue))
                    classes_found[ytrue] = True
            elif num_elements < n_classes and len(x0_list) < num_elements:
                x0_list.append((x0,ytrue))
                classes_found[ytrue] = True
            elif (len(x0_list) >= num_elements) :
                return x0_list
        
    return x0_list



def create_folder_benchmark_(
        data_modele: str,
        optimization_models: str,
        parametres_reseau: Dict,
        parametres_optimization: Dict,
        parametres_gurobi: Dict,
        x0_list: List[Tuple[List[float], int]],
        resultats: pd.DataFrame,
        just_change_bench_csv : bool = False, 
        option : str = None):
    """ Enregistre les résultats du run dans un dossier """
    
    n = parametres_reseau["n"]
    K = parametres_reseau["K"]
    epsilon = parametres_optimization["epsilon"]
    
    # Création du chemin de dossier
    folder_dir = f"datasets\{data_modele}\Benchmark\{data_modele}_epsilon={epsilon}_taille={len(x0_list)}"
    if not os.path.exists(folder_dir) and not just_change_bench_csv:
        os.makedirs(folder_dir)

    # Création du nom du fichier de résultats
    resultats_file_name = f"{data_modele}_epsilon={epsilon}_neurones={sum(n[1:K+1])}_taille={len(x0_list)}_{option}_benchmark.csv"
    resultats_file_path = os.path.join(folder_dir, resultats_file_name)
    
    # Supprimer le fichier de résultats s'il existe déjà
    if just_change_bench_csv and os.path.exists(resultats_file_path):
        print("resultats file path : ", resultats_file_path)
        os.remove(resultats_file_path)
        print(f"Fichier existant supprimé : {resultats_file_path}")
    
    # Sauvegarde des données
    list_file_path = os.path.join(folder_dir, 'x0_liste.json')
    with open(list_file_path, 'w') as list_file:
        json.dump(x0_list, list_file, indent=4)
        print(f"Liste enregistrée dans : {list_file_path}")

    dict_file_path_reseau = os.path.join(folder_dir, 'Parametres_reseau.json')
    with open(dict_file_path_reseau, 'w') as dict_file:
        json.dump({"Data_modele": data_modele,
                   "Optimization_models": optimization_models} |    
                  parametres_reseau, dict_file, indent=4)
        print(f"Dictionnaire des paramètres réseau enregistré dans : {dict_file_path_reseau}")

    dict_file_path_optimization = os.path.join(folder_dir, 'Parametres_optimization.json')
    with open(dict_file_path_optimization, 'w') as dict_file:
        json.dump(parametres_optimization, dict_file, indent=4)
        print(f"Dictionnaire des paramètres optimisation enregistré dans : {dict_file_path_optimization}")

    dict_file_path_gurobi = os.path.join(folder_dir, 'Parametres_gurobi.json')
    with open(dict_file_path_gurobi, 'w') as dict_file:
        json.dump(parametres_gurobi, dict_file, indent=4)
        print(f"Dictionnaire des paramètres de gurobi enregistré dans : {dict_file_path_gurobi}")

    # Enregistrement du fichier CSV de résultats
    if resultats != []:
        resultats_df = pd.DataFrame(resultats)
        print("resultats : ", resultats_df)
        resultats_df.to_csv(resultats_file_path, index=False)
        print(f"Résultats enregistrés dans : {resultats_file_path}")