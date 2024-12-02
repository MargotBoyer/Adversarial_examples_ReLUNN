# Model from Fischetti article

import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
import sys
from typing import List, Dict


from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_objectif import add_objective_dist
from GUROBI_variables import(
    add_variable_z,
    add_variable_sigma,
    add_variable_s,
    add_variable_d
)
from GUROBI_contraintes import(
    add_distance_constraints,
    add_hidden_layer_constraints_with_s,
    add_adversarial_constraints_ycible,
    add_last_layer,
    add_somme_beta_superieure_1
)

#from MILP.Code_Margot.Models_gurobi.Model_functions import apply_feasRelax

# Ce code cherche à trouver un exemple de classe ycible à partir de la donnee x0
# Cette fois ci la classe predite est obligatoirement ycible
# En minimisant la distance à x0 (tout en imposant que cette distance soit inférieure à epsilon)


def solveFischetti_Objdist(
    K : int, 
    n : List[int], 
    x0 : List[float], 
    ycible : int, 
    U : List[List[float]], 
    L : List[List[float]], 
    W : List[List[List[float]]], 
    b : List[List[float]], 
    rho : float, 
    epsilon : float, 
    relax : bool, 
    parametres_gurobi : Dict, 
    verbose : bool = False
):

    # Create a new model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Fischetti", env=env)
    adapt_parametres_gurobi(m,parametres_gurobi)


    z = add_variable_z(m,K,n,L)
    s = add_variable_s(m,K,n)
    sigma = add_variable_sigma(m,K,n,relax)
    d = add_variable_d(m,K,n)

    # -------------------- Fonction objectif --------------------#

    add_objective_dist(m, d, n)

    # -------------------- Contraintes ---------------------------#
    # Contraintes ReLU pour les hidden layers
    add_hidden_layer_constraints_with_s(m,z,s,sigma,K,n,W,b,U,L)

    # Contrainte derniere couche sans ReLU

    # Contraintes sur la boule initiale (definition de la distance d)
    add_distance_constraints(m, z, d, x0, epsilon, n)

    # Contraintes definissant un exemple adverse
    add_adversarial_constraints_ycible(m,z, K, n, W, b, rho, ycible)

    m.write("Models_gurobi/lp/fisch_dist.lp")
    # m.setParam("LogFile", "ReLUFischettiObjdist.log")
    m.optimize()
    print("Nombre de contraintes dans le modèle:", m.NumConstrs)

    # m.printAttr(['lb', 'ub'])
    # print (m.display())

    Sol = []
    status = -1
    opt = None
    time_execution = m.runtime
    nb_nodes = m.NodeCount
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        if verbose:
            print("Valeur optimale : ", round(opt, 4))
        if verbose:
            print("CPU time :", time_execution)
        for j in range(n[0]):
            Sol.append(z[0, j].X)
        status = 1
        if verbose:
            print("SOL : ", Sol)
        classes_finales = []
        alphas = []
        for k in range(1,K):
            alpha_k = []
            for j in range(n[k]):
                alpha_k.append(sigma[k,j].X)
            alphas.append(alpha_k)

        for j in range(n[K]):
            classes_finales.append(z[K, j].X)
        if verbose:
            print("Classes finales: ", classes_finales)
            print("alphas : ", alphas)

    elif m.Status == GRB.INFEASIBLE:
        if verbose : 
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/fisch_dist.ilp")
        status = 3
        if verbose : 
            print("L'IIS a été écrit dans 'model.ilp'")
    elif m.status == GRB.TIME_LIMIT:
        print("Temps limite atteint, récupération de la meilleure solution réalisable")
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            for j in range(n[0]):
                Sol.append(z[0, j].X)
            return Sol, m.ObjBound, 2, time_execution, {"Number_Nodes" : nb_nodes}
        status = 2
        
    else:
        if verbose :
            print("Statut modele : ", m.Status)

    return Sol, opt, status, time_execution, {"Number_Nodes" : nb_nodes}



def test():
    K=3
    W=[[[0.3,-0.2],[-0.15,-0.5],[0.3,0.5]],[[0.3,-0.2,-0.15,0.6],[0.3,-0.2,-0.15,0.6]] ,[[0.3,-0.2,0.4],[-0.1,-0.15,0.6],[-0.5,-0.2,-0.2],[0.3,0.5,0.2]]]
    b=[[-0.2,0.1],[0.3,-0.2,0.3,-0.1],[0.2,-0.5,0.1]]
    n=[3,2,4,3]
    x0=[0.6,-0.3,0.4]
    u=15
    l=-15
    y0=0
    # # K=3
    # # W=[[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]],[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]],[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]]]
    # # b=[[-0.2,0.1,0.4],[0.3,-0.2,0.3],[-0.1,0.2,-0.5]]
    # # n=[3,3,3,3]
    # # x0=[0.6,-0.3,0.4]
    # # u=15
    # # l=-15
    # # y0=2
    # # # print(K)
    # # # print(W)
    # # # print(b)
    # # # print(x0)
    # # # print(l)
    # # # print(u)
    # # # print(y0)
    # # # print(n)
    U= []
    for k in range(K+1):
        nvline = []
        for j in range(n[k]):
            nvline.append(u)
        U.append(nvline)
    #print(uX)
    L= []
    for k in range(K+1):
        nvline = []
        for j in range(n[k]):
            nvline.append(l)
        L.append(nvline)

    rho = 0.03
    epsilon = 1
    relax = 1
    verbose = 1
    Sol,opt,status,time, dic_nb_nodes = solveFischetti_Objdist(K,n,x0,y0,U,L,W,b,rho,epsilon,relax,parametres_gurobi,verbose)
    #Sol = solveFischetti(K,n,x0,U,L,W,b,y0,1,0)
    print("Sol : ", Sol)
    print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])



if __name__ == "__main__":
    test()
