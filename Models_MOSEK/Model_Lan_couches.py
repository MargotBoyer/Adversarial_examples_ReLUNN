import mosek
import numpy as np
import time
import os
import sys
from typing import Dict, List

from MOSEK_objective import (
    objective_function_diff_ycible
)
from MOSEK_outils import(
    reconstitue_matrice,
    adapte_parametres_mosek,
    affiche_matrice
)
from MOSEK_contraintes_adversariales import(
    contrainte_exemple_adverse_beta_u,
    contrainte_exemple_adverse_somme_beta_superieure_1,
    contrainte_beta_discret
)
from MOSEK_contraintes_passage_couches import (
    contrainte_quadratique_bornes,
    contrainte_ReLU_Mix,
    contrainte_derniere_couche_lineaire
)
from MOSEK_contraintes_generiques import (
    contrainte_boule_initiale,
    contrainte_premier_terme_egal_a_1
)
from MOSEK_contraintes_par_couches import(
    contrainte_recurrence_matrices_couches
)
from MOSEK_coupes import(
    contrainte_McCormick_zk2,
    contrainte_Mc_Cormick_betai_betaj,
    coupes_RLT_LAN,
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("path sys : ", sys.path)

from reseau_train import Reseau  # noqa: E402

inf = 10e5




def solve_Lan_couches(
    cert,
    coupes: Dict[str, bool], 
    ycible : int,
    titre : str,
    verbose : bool =True
):
    with mosek.Env() as env:
        with env.Task() as task:
            def streamprinter(text):
                if verbose : 
                    sys.stdout.write(text)
                    sys.stdout.flush()
                else :
                    pass  # N'affiche aucun log du solveur

            task.set_Stream(mosek.streamtype.log, streamprinter)
            adapte_parametres_mosek(task)
            numvar = 0  # Variables "indépendantes" -rien ici
            numcon = sum(cert.n[1:cert.K]) * 3 + 3 * cert.n[cert.K] + sum(cert.n[:cert.K]) + cert.n[0] + cert.K + 3 * cert.n[cert.K] - 2
            # Ajout contrainte sur les zk^2
            if coupes["zk2"]:
                numcon += (2 * sum(cert.n[1:cert.K]) + 3 * cert.n[0])
            # Ajout contrainte RLT Lan
            if coupes["RLTLan"]:
                numcon += (3 * sum(cert.n[k]*cert.n[k+1] for k in range(cert.K)) + sum(cert.n[1:cert.K]) 
                           + 2 * sum((cert.n[k])*(cert.n[k]-1)//2 for k in range(1,cert.K)))

            
            if verbose :
                print('Nombre de contraintes initialisées : ', numcon)

            task.appendcons(numcon)

            # Ajout des variables semi-définies du problème : ici les K-1 matrices représentant les z 
            for k in range(cert.K):
                task.appendbarvars([1 + cert.n[k] + cert.n[k+1]])
            # Ajout des variables "indépendantes" de la matrice sdp (ici 0 variable)
            task.appendvars(numvar)

            # ------------ FONCTION OBJECTIF ------------------------------------
            objective_function_diff_ycible(task, cert.K, cert.n, cert.y0, ycible, numvar,par_couches=True)
            # --------------------------------------------------------------------

            # ------------ CONTRAINTES RELU  ------------------------------------
            num_contrainte = 0

            # ***** Contrainte 1 :  zk+1 >= Wk zk + bk ********************
            # ***** Contrainte 2 :  zk+1 x (zk+1 - Wk zk - bk)  == 0  *****
            num_contrainte = contrainte_ReLU_Mix(task,cert.K,cert.n,cert.W,cert.b,num_contrainte,par_couches=True, 
                                                 neurones_actifs_stables=cert.neurones_actifs_stables,
                                                 neurones_inactifs_stables=cert.neurones_inactifs_stables)

            # ***** Contrainte 3 :   zK+1 == WK zK + bK *****
            num_contrainte = contrainte_derniere_couche_lineaire(task,cert.K,cert.n,cert.W,cert.b,num_contrainte,par_couches=True)

            # ***** Contrainte 4 :   Bornes sur les zkj hidden layers   *****
            num_contrainte = contrainte_quadratique_bornes(task, cert.K, cert.n, cert.L, cert.U, cert.x0, cert.epsilon, num_contrainte, par_couches=True)

            # ***** Contrainte 5 :   x - epsilon < z0 < x + epsilon  *****
            num_contrainte = contrainte_boule_initiale(task,cert.n,cert.x0,cert.epsilon,cert.U,cert.L,num_contrainte)

            # ***** Contrainte 6 : Pk[zk+1] == Pk+1[zk+1] ****************************
            num_contrainte = contrainte_recurrence_matrices_couches(task,cert.K,cert.n,num_contrainte)
            if verbose : 
                print("Nombre de contraintes après contrainte 6", num_contrainte)

            # Contrainte 7 : X00 = 1 (Le premier terme de la matrice variable est 1)
            print("cert.K - 1 : ", cert.K-1)
            num_contrainte = contrainte_premier_terme_egal_a_1(task,cert.K,cert.K,num_contrainte)
            if verbose : 
                print("Nombre de contraintes après XOO = 1 : ", num_contrainte)
            # ***********************************************
            # ************ COUPES ***************************
            # ***********************************************
            # Contrainte 8 : Bornes sur zk^2 
            if coupes["zk2"]:
                num_contrainte = contrainte_McCormick_zk2(task, cert.K, cert.n, cert.x0, cert.U, cert.L, cert.epsilon, 
                                                          num_contrainte,par_couches=True, neurones_actifs_stables=cert.neurones_actifs_stables,
                                                          neurones_inactifs_stables=cert.neurones_inactifs_stables)
                print("num contrainte apres zk2 : ", num_contrainte)
            # Contrainte 9 : Contraintes RLT
            if coupes["RLTLan"]:
                num_contrainte = coupes_RLT_LAN(task, cert.K, cert.n, cert.W, cert.b, cert.x0, cert.epsilon,
                                                cert.L, cert.U,num_contrainte,par_couches=True)
                print("num contrainte apres RLT : ", num_contrainte)

            if verbose : 
                print("Nombre de contraintes ajoutées dans le modèle : ", num_contrainte)
            # Configurer le solveur pour une optimisation
            task.putobjsense(mosek.objsense.minimize)

            task.writedata("Models_MOSEK/ptf/Model_Lan_couches.ptf")

            # Résoudre le problème
            start_time = time.time()
            task.optimize()
            end_time = time.time()

            time_execution = end_time - start_time

            # Extraire la solution
            solsta = task.getsolsta(mosek.soltype.itr)

            status = -1
            Sol = []
            num_iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
            print("Lan couches Solsta : ", solsta)
            if solsta == solsta.optimal:
                z_sol = task.getbarxj(mosek.soltype.itr, 0)
                z_0 = reconstitue_matrice(1 + cert.n[0] + cert.n[1], z_sol)
                affiche_matrice(cert,z_0,"Lan_couches_SDP",titre,coupes,nom_variable=f"z_{0}")
                # Assuming the optimization succeeded read solution
                for i in range(1,cert.K):
                    z_sol_i = task.getbarxj(mosek.soltype.itr, i)
                    z_i = reconstitue_matrice(1 + cert.n[i] + cert.n[i+1], z_sol_i)
                    affiche_matrice(cert,z_i,"Lan_couches_SDP",titre,coupes,nom_variable=f"z_{i}")


                # Obtenir la valeur du problème primal
                primal_obj_value = task.getprimalobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème primal: {primal_obj_value}")

                # Obtenir la valeur du problème dual
                dual_obj_value = task.getdualobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème dual: {dual_obj_value}")

                
                for j in range(cert.n[0]):
                    Sol.append(z_sol[j + 1])
                status = 1
                return Sol, primal_obj_value,status,  time_execution, {"Nombre_iterations" : num_iterations}

            elif solsta == solsta.dual_infeas_cer or solsta == solsta.prim_infeas_cer:
                if verbose : 
                    print("Primal or dual infeasibility certificate found.\n")
                status = 3
            elif solsta == solsta.unknown:
                if verbose : 
                    print("Unknown solution status")
            else:
                if verbose :  
                    print("Other solution status")

            return Sol, None, status, time_execution,  {"Nombre_iterations" : num_iterations}


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

    uns = [1, 1, 1]
    u = 150
    lb = -15

    Res = Reseau(K, n, W_reverse, b)

    x0 = [0.6, -0.3, 0.4]
    y0 = Res.retourne_label(x0)

    ycible = 1
    U = [[u] * taille for taille in n]
    L = [[lb] * taille for taille in n]
    coupes: Dict[str, bool] = {"zk^2": True, "betai*betaj": False, "RLT_Lan" : False}
    rho = 0.01
    epsilon = 5
    relax = 0
    ### Résoudre le problème QP
    solution, primal_obj_value, status, time_execution, dic_num_iterations = solve_Lan_couches(
        K, n, x0, y0, ycible, U, L, W_reverse, b, epsilon, rho, coupes, verbose=1
    )
    print("Solution:", solution)
    print("Nombre d'itérations : ", dic_num_iterations["Nombre_iterations"])
    if solution is not None:
        sortie = Res(solution, True)
        print("Sortie du réseau pour la solution calculée: ", sortie)
        predicted_label = Res.retourne_label(solution)
        print("Label retourné: ", predicted_label)


if __name__ == "__main__":
    test()
