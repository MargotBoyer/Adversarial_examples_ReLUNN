#import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict

from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_objectif import add_objective_dist
from GUROBI_variables import(
    add_variable_sigma,
    add_variable_z,
    add_variable_d
)
from GUROBI_contraintes import (
    add_distance_constraints,
    add_hidden_layer_constraints_with_sigma_linear_Glover,
    add_adversarial_constraints_ycible,
    add_somme_beta_superieure_1
)


def solveGlover(
        K : int, 
        n : List[int], 
        x0 : List[float], 
        ycible : int, 
        U : List[List[float]], 
        L : List[List[float]], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        epsilon : float, 
        rho : float, 
        relax : bool, 
        parametres_gurobi : Dict,
        verbose : bool = False):
    # Create a new model
    env = gp.Env(empty=True)
    if not verbose : 
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Glover", env=env)
    adapt_parametres_gurobi(m,parametres_gurobi)
 
    z = add_variable_z(m,K,n,L)
    d = add_variable_d(m,K,n)
    sigma = add_variable_sigma(m,K,n,relax)

    # -------------------- Fonction objectif --------------------#
    add_objective_dist(m,d,n)
    # -------------------- Contraintes ---------------------------#

    # Contraintes sur la boule autour de la donnée initiale
    # Contraintes definissant la distance a la donnee initiale
    add_distance_constraints(m,z,d,x0,epsilon,n)

    # Contraintes hidden layers avec ReLU
    add_hidden_layer_constraints_with_sigma_linear_Glover(m,z,sigma,W,b,K,n,U,L)

    # Contrainte derniere couche sans ReLU

    # Contraintes definissant un exemple adverse
    add_adversarial_constraints_ycible(m,z,K,n,W,b,rho,ycible)

    
    # m.printStats()
    m.write("Models_gurobi/lp/glover_obj_dist.lp")
    # m.setParam("LogFile", "ReLUGloverObjdist.log")
    m.optimize()
    # print (m.display())

    Sol = []
    status = -1
    time_execution = m.runtime
    opt = None
    nb_nodes = m.NodeCount
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        if verbose : 
            print("Valeur optimale : ", round(opt, 4))
            print("CPU time :", m.runtime)

        for j in range(n[0]):
            Sol.append(z[0, j].X)

        if False:
            for k in range(K + 1):
                vector_post_activation = []
                alphas = []
                if k >= 1:
                    vecteur_preactivation = []
                for j in range(n[k]):
                    vector_post_activation.append(z[k, j].X)
                    if k >= 1:
                        vecteur_preactivation.append(
                            gp.quicksum(
                                W[k - 1][i][j] * z[k - 1, i].X for i in range(n[k - 1])
                            )
                            + b[k - 1][j]
                        )
                        print(f" alpha {k},{j} ", sigma[k, j])
                if k >= 1:
                    print(f"Vecteur preactivation {k} : ", vecteur_preactivation)
                    # print(f"Alpha {k} : ", alpha)
                print(f"SOL couche {k} : ", vector_post_activation)

        status = 1
        if verbose:
            print("SOL : ", Sol)
        classes_finales = []
        for j in range(n[K]):
            classes_finales.append(z[K, j].X)
        if verbose:
            print("Classes finales: ", classes_finales)

        return Sol, opt, status, time_execution, {"Number_Nodes" : nb_nodes}

    elif m.Status == GRB.INFEASIBLE:
        if verbose : 
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/glover_obj_dist.ilp")
        status = 3   
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
    # Example usage:
    K = 3
    W = [[[0.3,-0.2],[-0.15,-0.5],[0.3,0.5]],[[-0.3,0.1,0.15,0.8],[0.3,-0.2,-0.15,0.6]], [[0.3,-0.2,0.4],[-0.1,-0.15,0.6],[-0.5,-0.2,-0.2],[0.3,0.5,0.2]]]
    W_reverse = [ [ [couche[i][j]  for i in range(len(couche))]  for j in range(len(couche[0]))  ] for couche in W]
    print("W reverse : ", W_reverse)
    b = [[-0.2,0.1],[0.3,-0.2,0.3,-0.1],[0.2,-0.5,0.1]]
    n = [3,2,4,3]

    x0=[0.6,-0.3,0.4]
    #x0=[8,3,1]
    ub=15
    lb=-15
    #ytrue=2
    ycible=1

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

    epsilon =10
    rho = 0.1
    relax = 0

    Sol, opt, status, time_execution, dic_nb_nodes = solveGlover(K,n,x0,ycible,U,L,W,b,epsilon,rho,relax, parametres_gurobi)
    print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])


if __name__ == "__main__":
    test()

