#import tupledict
import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from typing import List, Dict



from GUROBI_outils import adapt_parametres_gurobi, parametres_gurobi
from GUROBI_objectif import add_objective_diff
from GUROBI_variables import (
    add_variable_z,
    add_variable_beta,
    add_variable_sigma
)
from GUROBI_contraintes import (
    add_adversarial_constraints,
    add_initial_ball_constraints,
    add_hidden_layer_constraints_with_sigma_quad,
    add_somme_beta_superieure_1
)



def solveFprG_quad(
    cert, 
    relax : bool,
    titre : str, 
    verbose : bool = True
):
    # Create a new model
    env = gp.Env(empty=True)
    if verbose:
        env.setParam("OutputFlag", 1)
    else:
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("FprG_quad", env=env)
    adapt_parametres_gurobi(m,parametres_gurobi)

    z = add_variable_z(m, cert.K, cert.n, cert.L)
    beta = add_variable_beta(m, cert.K, cert.n, relax)
    sigma = add_variable_sigma(m, cert.K, cert.n, relax)

    # -------------------- Fonction objectif --------------------#
    add_objective_diff(m, z, cert.W, cert.b, cert.K, cert.n, cert.y0)
    # -------------------- Contraintes ---------------------------#

    # Contraintes sur la boule autour de la donnée initiale
    add_initial_ball_constraints(m, z, cert.x0, cert.epsilon, cert.n, cert.L[0], cert.U[0])

    # Contraintes hidden layers avec ReLU
    add_hidden_layer_constraints_with_sigma_quad(m, z, sigma, cert.W, cert.b, cert.K, 
                                                 cert.n, cert.U, cert.L)

    # Contrainte derniere couche sans ReLU

    # Contraintes definissant un exemple adverse
    add_adversarial_constraints(m, z, beta, cert.W, cert.b, cert.U, cert.K, 
                                cert.n, cert.y0, cert.rho)
    add_somme_beta_superieure_1(m,beta,cert.K,cert.n,cert.y0)

    # m.printStats()
    m.write("Models_gurobi/lp/FprG_quad.lp")
    # m.setParam("LogFile", "ReLUGloverObjdist.log")
    m.optimize()
    # print (m.display())
    print("Nombre de contraintes dans le modèle:", m.NumConstrs)

    Sol = []
    status = -1
    time_execution = m.runtime
    opt = None
    nb_nodes = m.NodeCount
    if m.Status == GRB.OPTIMAL:
        opt = m.ObjVal
        if verbose:
            print("Valeur optimale : ", round(opt, 4))
            print("CPU time :", m.runtime)

        for j in range(cert.n[0]):
            Sol.append(z[0, j].X)

        status = 1
        if verbose:
            print("SOL : ", Sol)
        classes_finales = []
        for j in range(cert.n[cert.K]):
            classes_finales.append(z[cert.K, j].X)
        if verbose:
            print("Classes finales: ", classes_finales)

        return Sol, opt, status, time_execution, {"Number_Nodes" : nb_nodes}

    elif m.Status == GRB.INFEASIBLE:
        if verbose:
            print("Modele infaisable !")
        status = 3
    elif m.status == GRB.TIME_LIMIT:
        print("Temps limite atteint, récupération de la meilleure solution réalisable")
        if m.SolCount > 0:
            print("Solution réalisable disponible")
            for j in range(cert.n[0]):
                Sol.append(z[0, j].X)
            return Sol, m.ObjBound, 2, time_execution, {"Number_Nodes" : nb_nodes}
        status = 2
    else:
        if verbose:
            print("Statut modele : ", m.Status)
    return Sol, opt, status, time_execution, {"Number_Nodes" : nb_nodes}


def test():
    # Example usage:
    K = 3
    W = [
        [[0.3, -0.2], [-0.15, -0.5], [0.3, 0.5]],
        [[-0.3, 0.1, 0.15, 0.8], [0.3, -0.2, -0.15, 0.6]],
        [[0.3, -0.2, 0.4], [-0.1, -0.15, 0.6], [-0.5, -0.2, -0.2], [0.3, 0.5, 0.2]],
    ]
    W_reverse = [
        [[couche[i][j] for i in range(len(couche))] for j in range(len(couche[0]))]
        for couche in W
    ]
    b = [[-0.2, 0.1], [0.3, -0.2, 0.3, -0.1], [0.2, -0.5, 0.1]]
    n = [3, 2, 4, 3]

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


    Sol, opt, status, time_execution, dic_nb_nodes = solveFprG_quad(K, n, x0, ytrue, U, L, W, b, epsilon, epsilon_adv, relax, parametres_gurobi)
    print("nb nodes : ", dic_nb_nodes["Number_Nodes"])




if __name__ == "__main__":
    test()
