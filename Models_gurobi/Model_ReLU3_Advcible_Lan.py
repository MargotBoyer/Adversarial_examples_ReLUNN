import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict


from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_objectif import add_objective_diff_ycible

from GUROBI_variables import(
    add_variable_z
)
from GUROBI_contraintes import(
    add_initial_ball_constraints,
    add_hidden_layer_constraints_quad,
    add_quadratic_bounds_on_all_layers_constraints
)


def solve_Lan_quad(
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
        parametres_gurobi : Dict,
        verbose : bool = False
        ):
    env = gp.Env(empty=True)
    if verbose : 
        env.setParam("OutputFlag",1)
    else : 
        env.setParam("OutputFlag",0)
    # env.setParam("DualReductions",0)
    env.start()
    m = gp.Model("Lan_quad", env=env)
    adapt_parametres_gurobi(m,parametres_gurobi)

   
    z = add_variable_z(m, K, n, L)

    #-------------------- Fonction objectif --------------------#

    add_objective_diff_ycible(m, z, W, b, K, n, ytrue, ycible)

    #-------------------- Contraintes ---------------------------#
    # Contraintes de bornes quadratiques sur la boule initiale et sur les couches internes
    add_quadratic_bounds_on_all_layers_constraints(m,z,x0,K,U,L,n,epsilon)

    # Contraintes couches internes avec ReLU
    add_hidden_layer_constraints_quad(m,z,W,b,K,n)

    #m.printStats()
    m.write("Models_gurobi/lp/Lan_quad.lp")
    
    #m.setParam("LogFile", "Lan_quad.log")
    m.optimize()
    #print (m.display()) 
    
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
        for j in range(n[K]):
            if j== ytrue  :
                print(f"Sortie pour j=ytrue={j} : {gp.quicksum(W[K-1][i][ytrue] * z[K-1,i].X for i in range(n[K-1])) + b[K-1][ytrue]}")
            else :
                print(f"Sortie pour j={j}] : {gp.quicksum(W[K-1][i][j] * z[K-1,i].X for i in range(n[K-1])) + b[K-1][j]}")
    
   
        classes_finales = []
        for j in range(n[K]):
            classes_finales.append(z[K,j].X)
        if verbose : 
            print("Classes finales: ", classes_finales)

        return Sol,opt,status,time_execution, {"Number_Nodes" : nb_nodes}
    
    elif m.Status == GRB.INFEASIBLE:
        if verbose :
            print("Modele infaisable !")
        m.computeIIS()
        m.write("Models_gurobi/lp/Lan_quad.ilp")
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
    return Sol,opt,status,0, {"Number_Nodes" : nb_nodes}


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
    relax = 0
    Sol, opt , status, time_ex, dic_nb_nodes = solve_Lan_quad(K,n,x0,ytrue,ycible,U,L,W,b,epsilon,parametres_gurobi,verbose=False)
    print("Sol : ", Sol)
    print("Nombre de noeuds : ", dic_nb_nodes["Number_Nodes"])




if __name__ == "__main__":
    test()
