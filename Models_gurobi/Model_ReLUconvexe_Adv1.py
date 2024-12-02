import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
import sys
from typing import List, Dict


from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_objectif import add_objective_diff
from GUROBI_variables import(
    add_variable_beta,
    add_variable_sigma,
    add_variable_z,
    add_variable_s
)
from GUROBI_contraintes import(
    add_initial_ball_constraints,
    add_hidden_layers_ReLU_convex_relaxation,
    add_adversarial_constraints,
    add_somme_beta_superieure_1
)



def solve_ReLUconvexe_Adv1(
        K : int, 
        n : List[int], 
        x0 : List[float], 
        ytrue : int, 
        U : List[List[float]], 
        L : List[List[float]], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        epsilon_adv : float, 
        epsilon : float, 
        relax : bool, 
        parametres_gurobi : Dict, 
        verbose : bool = False
        ):
    # Create a new model
    env = gp.Env(empty=True)
    if not verbose : 
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Fischetti", env=env)
    adapt_parametres_gurobi(m, parametres_gurobi)
    

    z = add_variable_z(m,K,n,L)
    beta = add_variable_beta(m, K, n, relax)
    s = add_variable_s(m,K,n)
    sigma = add_variable_sigma(m,K,n,relax)

    # -------------------- Fonction objectif --------------------#

    add_objective_diff(m,z,W,b,K,n,ytrue)

    # -------------------- Contraintes ---------------------------#
    # Contrainte sur la boule autour de la donnee initiale
    add_initial_ball_constraints(m,z,x0,epsilon,n, L[0], U[0])

    # Contraintes hidden layers avec ReLU
    add_hidden_layers_ReLU_convex_relaxation(m,z,K,n,W,b,U,L)

    # Contrainte derniere couche sans ReLU

    # Contraintes definissant un exemple adverse
    add_adversarial_constraints(m,z,beta,W,b,U,K,n,ytrue,epsilon_adv)
    add_somme_beta_superieure_1(m,beta,K,n,ytrue)
    

    m.write("Models_gurobi/lp/ReLUconvexe_Adv1.lp")
    m.setParam("LogFile", "ReLUconvexe_Adv1.log")
    m.optimize()
    # print (m.display())
    print("Nombre de contraintes dans le modèle:", m.NumConstrs)

    Sol = []
    status = -1
    opt = None
    nb_nodes = m.NodeCount
    time_execution = m.runtime
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        if verbose:
            print("Valeur optimale : ", round(opt, 4))
        if verbose:
            print("CPU time :", m.runtime)

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
            print("Alphas : ", alphas)
    elif m.Status == 3:
        status = 3
        if verbose :
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/ReLUconvexe_Adv1.ilp")
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
    ub=15000
    lb=-15000
    y0=0
    # K=3
    # W=[[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]],[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]],[[0.3,-0.2,0.4],[-0.15,0.6,-0.5],[-0.2,0.3,0.5]]]
    # b=[[-0.2,0.1,0.4],[0.3,-0.2,0.3],[-0.1,0.2,-0.5]]
    # n=[3,3,3,3]
    # x0=[0.6,-0.3,0.4]
    # u=15
    # l=-15
    # y0=2
    # # print(K)
    # # print(W)
    # # print(b)
    # # print(x0)
    # # print(l)
    # # print(u)
    # # print(y0)
    # # print(n)
    U= []
    for k in range(K+1):
        nvline = []
        for j in range(n[k]):
            nvline.append(ub)
        U.append(nvline)
    #print(uX)
    L= []
    for k in range(K+1):
        nvline = []
        for j in range(n[k]):
            nvline.append(lb)
        L.append(nvline)

    rho = 0.03
    epsilon = 10
    relax = 1
    verbose = 1
    Sol,opt,status,time_exe,dic_nb_nodes = solve_ReLUconvexe_Adv1(K,n,x0,y0,U,L,W,b,rho,epsilon,relax,parametres_gurobi,verbose)
    #Sol = solveFischetti(K,n,x0,U,L,W,b,y0,1,0)
    print("Sol : ", Sol)
    print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])



if __name__ == "__main__":
    test()
