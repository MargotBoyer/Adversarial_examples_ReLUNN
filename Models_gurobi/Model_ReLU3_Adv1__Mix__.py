#import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict

from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_contraintes import (
    add_hidden_layer_constraints_quad, 
    add_initial_ball_constraints,
    add_adversarial_constraints,
    add_somme_beta_superieure_1)
from GUROBI_variables import(
    add_variable_beta,
    add_variable_z
)
from GUROBI_objectif import(
    add_objective_diff
)


def solveMix(
        K : int,
        n : List[int],
        x0 : List[float],
        ytrue : int,
        ycible : int,
        U : List[List[float]],
        L : List[List[float]],
        W : List[List[List[float]]],
        b : List[List[float]],
        epsilon : float,
        epsilon_adv : float,
        relax : bool,
        parametres_gurobi : Dict,
        verbose : bool = False
        ):

    # Create a new model
    env = gp.Env(empty=True)
    if verbose : 
        env.setParam("OutputFlag",1)
    else :
        env.setParam("OutputFlag",0)
    env.setParam("DualReductions",0)
    env.start()
    m = gp.Model("Mix", env=env)
    adapt_parametres_gurobi(m, parametres_gurobi)

    
    z = add_variable_z(m, K, n, L)
    beta = add_variable_beta(m, K, n, relax)

    #-------------------- Fonction objectif --------------------#

    add_objective_diff(m, z, W, b, K, n, ytrue)

    #-------------------- Contraintes ---------------------------#
    # Contraintes sur la boule autour de la donnée initiale
    add_initial_ball_constraints(m, z, x0, epsilon, n, L[0], U[0])

    # Contraintes sur les bornes puor les hidden layers :
    
    # Contraintes hidden layers avec ReLU
    add_hidden_layer_constraints_quad(m, z, W, b, K, n)

    # Contrainte derniere couche sans ReLU

    # Contrainte définissant un exemple adversarial
    add_adversarial_constraints(m, z, beta, W, b, U, K, n, ytrue, epsilon_adv)
    add_somme_beta_superieure_1(m,beta,K,n,ytrue)

    #m.printStats()
    m.write("Models_gurobi/lp/Mix.lp")

    #m.setParam("LogFile", "ModelMIXquadObjdiff.log")
    m.optimize()
    #print (m.display()) 
    print("Nombre de contraintes dans le modèle:", m.NumConstrs)
    Sol = []
    status=-1
    opt=None
    time_execution = m.runtime
    nb_nodes = m.NodeCount
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        if verbose : 
            print("Valeur optimale : ",round(opt,4))
        if verbose : 
            print("CPU time :", time_execution)
        for j in range(n[0]):
            Sol.append(z[0,j].X)
        status=1
        if verbose : 
            print("Sol : ", Sol)
        # if False:
        #     for k in range(K+1):
        #         vector_post_activation = []
        #         if k>=1:
        #             vecteur_preactivation = []
        #         for j in range(n[k]):
        #             vector_post_activation.append(z[k,j].X)
        #             if k>=1: vecteur_preactivation.append(gp.quicksum(W[k-1][i][j] * z[k-1,i].X for i in range(n[k-1]))+ b[k-1][j])
        #         if k>=1: print(f"Vecteur preactivation {k} : ", vecteur_preactivation)
        #         print(f"SOL couche {k} : ", vector_post_activation)
        
   
        classes_finales = []
        for j in range(n[K]):
            classes_finales.append(z[K,j].X)
        betas = []
        for j in range(n[K]):
            betas.append(beta[j].X)
        print("Sol : ", Sol)
        print("Betas : ", betas)
        print("z premières couches : ", [[z[k,j].X for j in range(n[k])] for k in range(K)])
        for j in range(n[K]):
            if j== ytrue  :
                print(f"Sortie pour j=ytrue={j} : {gp.quicksum(W[K-1][i][ytrue] * z[K-1,i].X for i in range(n[K-1])) + b[K-1][ytrue]}")
            else :
                print(f"Sortie pour j={j}] : {gp.quicksum(W[K-1][i][j] * z[K-1,i].X for i in range(n[K-1])) + b[K-1][j]}")
        
        if verbose : 
            print("Classes finales: ", classes_finales)

        return Sol,opt,status,time_execution, {"Number_Nodes" : nb_nodes}
    
    elif m.Status == GRB.INFEASIBLE:
        if verbose :
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/Mix.ilp")
        status = 3
    elif m.status == GRB.TIME_LIMIT:
        print("Temps limite atteint, récupération de la meilleure solution réalisable")
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            for j in range(n[0]):
                Sol.append(z[0, j].X)
            return Sol, m.ObjBound, 2, time_execution, {"Number_Nodes" : nb_nodes}
        status = 2
    else :
        if verbose :
            print("Statut modele : ", m.Status)
    
    return Sol,opt,status,0,{"Number_Nodes" : nb_nodes}

def test():
    K=3
    W=[[[0.3,-0.2],[-0.15,-0.5],[0.3,0.5]],[[0.3,-0.2,-0.15,0.6],[0.3,-0.2,-0.15,0.6]] ,[[0.3,-0.2,0.4],[-0.1,-0.15,0.6],[-0.5,-0.2,-0.2],[0.3,0.5,0.2]]]
    b=[[-0.2,0.1],[0.3,-0.2,0.3,-0.1],[0.2,-0.5,0.1]]
    n=[3,2,4,3]
    x0=[0.6,-0.3,0.4]
    #x0=[0.8,-2,5]
    u=1500
    lb=-1500
    ytrue=0 # A CALCULER A PARTIR DE XO ET DU RESEAU DE NEURONES 
    ycible=2

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
            nvline.append(lb)
        L.append(nvline)

    epsilon =3
    epsilon_adv = 0.001
    relax = 1

    Sol, opt , status, time_ex, dic_nb_nodes = solveMix(K,n,x0,ytrue,ycible,U,L,W,b,epsilon,epsilon_adv,relax,parametres_gurobi,verbose=1)
    print("Sol : ", Sol)
    print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])



if __name__ == "__main__":
    test()
