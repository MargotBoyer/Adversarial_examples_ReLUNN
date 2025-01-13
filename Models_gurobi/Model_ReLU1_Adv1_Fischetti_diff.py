# Modele from Fischetti article in which we adapt the objective to find an adversarial example for all class (not only a targeted class)

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
    add_hidden_layer_constraints_with_s,
    add_adversarial_constraints,
    add_somme_beta_superieure_1
)


def solveFischetti_Objdiff(
       cert, 
       relax : bool,
       titre : str, 
       verbose : bool = True):
    """ Modele F' : M_1_1 """
    env = gp.Env(empty=True)
    if not verbose : 
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Fischetti_diff", env=env)
    adapt_parametres_gurobi(m, parametres_gurobi)
    
    
    z = add_variable_z(m,cert.K,cert.n,cert.L)
    beta = add_variable_beta(m, cert.K, cert.n, relax)
    s = add_variable_s(m,cert.K,cert.n)
    sigma = add_variable_sigma(m,cert.K,cert.n,relax)

    # -------------------- Fonction objectif --------------------#

    add_objective_diff(m,z,cert.W,cert.b,cert.K,cert.n,cert.y0)

    # -------------------- Contraintes ---------------------------#
    # Contrainte sur la boule autour de la donnee initiale
    add_initial_ball_constraints(m,z,cert.x0,cert.epsilon, cert.n, cert.L[0], cert.U[0])

    # Contraintes hidden layers avec ReLU
    add_hidden_layer_constraints_with_s(m, z, s, sigma, cert.K, cert.n, cert.W, cert.b, 
                                        cert.U, cert.L, cert.neurones_actifs_stables, cert.neurones_inactifs_stables)

    # Contrainte derniere couche sans ReLU

    # Contraintes definissant un exemple adverse
    add_adversarial_constraints(m,z,beta,cert.W,cert.b,cert.U,cert.K,cert.n,cert.y0,cert.rho)
    add_somme_beta_superieure_1(m,beta,cert.K,cert.n,cert.y0)
    

    m.write("Models_gurobi/lp/fisch_diff.lp")
    #m.setParam("LogFile", "ReLUFischettiObjdiff.log")
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

        for j in range(cert.n[0]):
            Sol.append(z[0, j].X)
        betas = []
        for j in range(cert.n[cert.K]):
            betas.append(beta[j].X
            )
        status = 1
        if verbose:
            print("SOL : ", Sol)
        print("betas : ", betas)
        classes_finales = []
        alphas = []
        for k in range(1,cert.K):
            alpha_k = []
            for j in range(cert.n[k]):
                alpha_k.append(sigma[k,j].X)
            alphas.append(alpha_k)

        for j in range(cert.n[cert.K]):
            classes_finales.append(z[cert.K, j].X)
        if verbose:
            print("Classes finales: ", classes_finales)
            print("Alphas : ", alphas)
    elif m.Status == 3:
        status = 3
        if verbose :
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/fisch_diff.ilp")
    elif m.status == GRB.TIME_LIMIT:
        print("Temps limite atteint, récupération de la meilleure solution réalisable")
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            for j in range(cert.n[0]):
                Sol.append(z[0, j].X)
            return Sol, m.ObjBound, 2, time_execution, {"Number_Nodes" : nb_nodes}
        status = 2
    else:
        if verbose :
            print("Statut modele : ", m.Status)
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            for j in range(cert.n[0]):
                Sol.append(z[0, j].X)

    
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
    # Sol,opt,status,time_exe,dic_nb_nodes = solveFischetti_Objdiff(K,n,x0,y0,U,L,W,b,rho,epsilon,relax,parametres_gurobi,verbose)
    # #Sol = solveFischetti(K,n,x0,U,L,W,b,y0,1,0)
    # print("Sol : ", Sol)
    # print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])



if __name__ == "__main__":
    test()
