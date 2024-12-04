import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict

import sys
import os

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



def solve_borne_sup_U(
    couche : int,
    neurone : int,
    K : int, 
    n : List[int], 
    x0 : List[float], 
    W : List[List[List[float]]], 
    b : List[List[float]], 
    L : List[List[float]],
    U : List[List[float]],
    epsilon : float, 
    relax : bool, 
    neurones_actifs_stables : List,
    neurones_inactifs_stables : List,
    parametres_gurobi : Dict, 
    verbose : bool = False
):
    env = gp.Env(empty=True)
    if verbose:
        env.setParam("OutputFlag", 1)
    else:
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Calcul_Borne_Sup_U", env=env)
    parametres_gurobi["TimeLimit"] = 300
    adapt_parametres_gurobi(m,parametres_gurobi)

    z = add_variable_z(m, K, n[:couche+1], L, couche, impose_positive=False)
    if couche > 0 :
        sigma = add_variable_sigma(m, couche-1, n, relax, neurones_actifs_stables + neurones_inactifs_stables)

    # -------------------- Fonction objectif --------------------#
    if couche > 0:
        add_objective_U_linear(m,z,W,b,n,couche,neurone)
    else :
        add_objective_U(m,z,couche,neurone)
    # -------------------- Contraintes ---------------------------#

    # Contraintes sur la boule autour de la donnée initiale
    if couche == 0:
        add_initial_ball_constraints(m, z, x0, epsilon, n, L[0], U[0], neurone)
    else : 
        add_initial_ball_constraints(m, z, x0, epsilon, n, L[0], U[0])
    
    
    if couche > 0 :
        # # Contrainte derniere couche sans ReLU
        # add_last_layer(m,z,couche,n,W,b,neurone)
        # Contraintes hidden layers avec ReLU 
        add_hidden_layer_constraints_with_sigma_linear_Glover(m, z, sigma, W, b, couche, n, U, L, 
                                                              neurones_actifs_stables, neurones_inactifs_stables)
        
    # m.printStats()
    m.write("Models_gurobi/lp/Bornes_max_U.lp")
    # m.setParam("LogFile", "ReLUGloverObjdist.log")
    m.optimize()
    # print (m.display())
    # print("Nombre de contraintes dans le modèle:", m.NumConstrs)   
    return retourne_valeurs_solutions_bornes(m,z,couche,neurone,n,"U",verbose=True)



def solve_borne_inf_L(
    couche : int,
    neurone : int,
    K : int, 
    n : List[int], 
    x0 : List[float], 
    W : List[List[List[float]]], 
    b : List[List[float]], 
    L : List[List[float]],
    U : List[List[float]],
    epsilon : float, 
    relax : bool, 
    neurones_actifs_stables : List,
    neurones_inactifs_stables : List,
    parametres_gurobi : Dict, 
    verbose : bool = False
):
    env = gp.Env(empty=True)
    if verbose:
        env.setParam("OutputFlag", 1)
    else:
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Calcul_Borne_Inf_L", env=env)
    parametres_gurobi["TimeLimit"] = 300
    adapt_parametres_gurobi(m,parametres_gurobi)

    z = add_variable_z(m, K, n[:couche+1], L, couche, impose_positive=False)
    if couche > 0:
        sigma = add_variable_sigma(m, couche-1, n, relax, neurones_actifs_stables + neurones_inactifs_stables)

    # -------------------- Fonction objectif --------------------#
    if couche > 0:
        add_objective_L_linear(m,z,W,b,n,couche,neurone)
    else :
        add_objective_L(m,z,couche,neurone)
    # -------------------- Contraintes ---------------------------#
    # Contraintes sur la boule autour de la donnée initiale
    if couche == 0:
        add_initial_ball_constraints(m, z, x0, epsilon, n, L[0], U[0], neurone)
    else : 
        add_initial_ball_constraints(m, z, x0, epsilon, n, L[0], U[0])
    
    
    if couche > 0 :
        # # Contrainte derniere couche sans ReLU
        # add_last_layer(m,z,couche,n,W,b,neurone)
        # Contraintes hidden layers avec ReLU 
        add_hidden_layer_constraints_with_sigma_linear_Glover(m, z, sigma, W, b, couche, n, U, L,
                                                              neurones_actifs_stables, neurones_inactifs_stables)
        

    # m.printStats()
    m.write("Models_gurobi/lp/Bornes_inf_L.lp")
    m.optimize()
    # # print (m.display())
    # print("Nombre de contraintes dans le modèle:", m.NumConstrs)
    L_neurone, status, time_execution, dic_nb_nodes = retourne_valeurs_solutions_bornes(m,z,couche,neurone,n,"L",verbose=True)
    return L_neurone, status, time_execution, dic_nb_nodes 



def Interval_Bound_Propagation(
    x0 : List[float], 
    K : int, 
    n : List[int], 
    W : List[List[List[float]]], 
    b : List[List[float]], 
    epsilon : float, 
    verbose : bool = False
):
    L = [[None for j in range(n[k])] for k in range(K+1)]
    L[0] = [(x0[j]-epsilon) for j in range(n[0])]

    U = [[None for j in range(n[k])] for k in range(K+1)]
    U[0] = [(x0[j]+epsilon) for j in range(n[0])]

    neurones_actifs_stables = []
    neurones_inactifs_stables = []

    for couche in range(1,K+1):
        for neurone in range(n[couche]):
            lb = 0
            ub = 0
            for i in range(n[couche-1]):
                if W[couche-1][i][neurone] > 0:
                    lb += W[couche-1][i][neurone] * L[couche-1][i]
                    ub += W[couche-1][i][neurone] * U[couche-1][i]
                else : 
                    lb += W[couche-1][i][neurone] * U[couche-1][i]
                    ub += W[couche-1][i][neurone] * L[couche-1][i]
            L[couche][neurone] = lb + b[couche-1][neurone]
            U[couche][neurone] = ub + b[couche-1][neurone]

            if lb > 0 : 
                neurones_actifs_stables.append((couche,neurone))
            elif ub < 0 : 
                neurones_inactifs_stables.append((couche,neurone))
    print("Resultats de l'IBP : ")
    print(f"L = {L}")
    print(f"U = {U}")
    return L, U, neurones_actifs_stables, neurones_inactifs_stables

    


def compute_FULL_U(x0, K, n, W, b, L, U, epsilon):
    U_new = []
    relax = 0
    for couche in range(0,K+1):
        U_couche= []
        for neurone in range(n[couche]):
            print(f"\n     Neurone {neurone} couche {couche} : ")
            U_neurone, status, time_execution, dic_nb_nodes = solve_borne_sup_U(couche, neurone, K, n, x0,  W, b, L, U, epsilon, relax, parametres_gurobi,verbose = True)      
            print(f"U = {U_neurone}")
            U_couche .append(U_neurone)
        U_new.append(U_couche)
    return U_new

def compute_FULL_L(x0, K, n, W, b, L, U, epsilon):
    L_new = []
    relax = 0
    for couche in range(0,K+1):
        L_couche= []
        for neurone in range(n[couche]):
            print(f"\n     Neurone {neurone} couche {couche} : ")
            U_neurone, status, time_execution, dic_nb_nodes = solve_borne_inf_L(couche, neurone, K, n, x0,  W, b, L, U, epsilon, relax, parametres_gurobi)
            print(f"L = {U_neurone}")
            L_couche .append(U_neurone)
        L_new.append(L_couche)
    return L_new


def compute_FULL_U_L(x0, K, n, W, b, L, U, epsilon, verbose = False, neurones_actifs_stables = [], neurones_inactifs_stables = []):
    L_new = []
    U_new = []

    relax = 0
    for feature in range(n[0]) :
        L_new.append(x0[feature] - epsilon)
        U_new.append(x0[feature] + epsilon)
    for couche in range(1,K+1):
        L_couche= []
        U_couche= []
        if couche==K:
            print("Adaptation du temps limite sur gurobi")
            parametres_gurobi["TimeLimit"] = 120
        for neurone in range(n[couche]):
            if (couche,neurone) in neurones_actifs_stables+ neurones_inactifs_stables:
                print(f"Le neurone {neurone} de la couche {couche} n'est pas traite car stable.")
                L_couche.append(L[couche][neurone])
                U_couche.append(U[couche][neurone])
                continue
            print(f"\n     Couche {couche}  neurone {neurone}  : ")
            print("Neurones actifs stables a ce jour : ", neurones_actifs_stables)
            print("Neurones inactifs stables a ce jour : ", neurones_inactifs_stables)
            L_neurone, status, time_execution, dic_nb_nodes = solve_borne_inf_L(couche, neurone, K, n, x0,  W, b, L, U, epsilon, 
                                                                                relax, neurones_actifs_stables, neurones_inactifs_stables, parametres_gurobi, verbose = verbose)            
            L_couche.append(L_neurone)
            print(f"L = {L_neurone}")

            U_neurone, status, time_execution, dic_nb_nodes = solve_borne_sup_U(couche, neurone, K, n, x0,  W, b, L, U, epsilon, 
                                                                                relax, neurones_actifs_stables, neurones_inactifs_stables, parametres_gurobi,verbose = verbose)
            print(f"U = {U_neurone}")
            U_couche.append(U_neurone)

            if (couche>0) & (couche<K) & (U_neurone < 0):
                print(f"Le neurone {neurone} de la couche {couche} est inactif stable.")
                neurones_inactifs_stables.append((couche,neurone))
            elif (couche>0) & (couche<K) & (L_neurone > 0) : 
                neurones_actifs_stables.append((couche,neurone))
                print(f"Le neurone {neurone} de la couche {couche} est actif stable.")
        L_new.append(L_couche)
        U_new.append(U_couche)

    print("Les neurones actifs stables sont : ", neurones_actifs_stables)
    print("Les neurones inactifs stables sont : ", neurones_inactifs_stables)
    return L_new, U_new, neurones_actifs_stables, neurones_inactifs_stables


def calcule_bornes_all_algorithms(x0,K,n,W,b,epsilon, verbose = False):
    print("IBP...")
    L_x0_IB, U_x0_IB, neurones_actifs_stables, neurones_inactifs_stables = Interval_Bound_Propagation(x0,K,n,W,b,epsilon,verbose=verbose)
    print("Neurones stables trouves apres IBP : ", neurones_actifs_stables+neurones_inactifs_stables)
    print("FULL...")
    L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables = compute_FULL_U_L(x0,K,n,W,b,L_x0_IB,U_x0_IB,epsilon,
                                                                                      verbose, neurones_actifs_stables, neurones_inactifs_stables)
    print("Neurones stables trouves apres FULL : ", neurones_actifs_stables+neurones_inactifs_stables)
    return L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables


def test():
    # Example usage:
    K = 3
    W = [
        [[0.3, -0.2], [-0.15, -0.5], [0.3, 0.5]],
        [[-0.15,0.6], [0.3,0.7]],
        [[-0.15], [0.3]]
        ]
    W_reverse = [
        [[couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))]
        for couche in W
    ]
    b = [[-0.2, 0.1], [-0.5,-1], [-0.3]]
    n = [3, 2, 2, 1]

    x0 = [0.6, -0.3, 0.4]
    ub = 15
    lb = -15
    ytrue = 2
    ycible = 1

    U = []
    for k in range(K + 1):
        nvline = []
        for j in range(n[k]):
            nvline.append(ub)
        U.append(nvline)
    L = []
    for k in range(K + 1):
        nvline = []
        for j in range(n[k]):
            nvline.append(lb)
        L.append(nvline)

    epsilon = 10
    epsilon_adv = 0.1
    relax = 0
    verbose = False
    # U = compute_FULL_U(x0, K, n, W, b, L, U, epsilon)
    # L = compute_FULL_L(x0, K, n, W, b, L, U, epsilon)
    L, U, _, _ = compute_FULL_U_L(x0, K, n, W, b, L, U, epsilon)
    print("Calcul exact des bornes : ")
    print("U : ", U)
    print("L : ", L)

    U, L, _, _ = Interval_Bound_Propagation(x0,K,n,W,b,epsilon)
    print("Interval Bound Propagation : ")
    print("U : ", U)
    print("L : ", L)

    L_x0, U_x0, neurones_actifs_stables, neurones_inactifs_stables = calcule_bornes_all_algorithms(x0,K,n,W,b,epsilon)


    # couche = 2
    # neurone = 0
    # L_neurone, status, time_execution, dic_nb_nodes = solve_borne_inf_L(couche, neurone, K, n, x0,  W, b, L, U, epsilon, relax, parametres_gurobi, verbose = True)
    # print(f"\n     Neurone {neurone} couche {couche} : ")
    # print(f"L = {L_neurone}")



if __name__ == "__main__":
    test()
