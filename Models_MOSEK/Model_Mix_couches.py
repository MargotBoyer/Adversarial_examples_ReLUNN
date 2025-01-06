import mosek
import numpy as np
import time
import os
import sys
from typing import Dict, List

from MOSEK_objective import (
    objective_function_diff_par_couches
)
from MOSEK_outils import(
    reconstitue_matrice,
    affiche_matrice
)
from MOSEK_contraintes_adversariales import(
    contrainte_exemple_adverse_beta_u,
    contrainte_exemple_adverse_somme_beta_superieure_1,
    contrainte_beta_discret,
)
from MOSEK_contraintes_passage_couches import (
    contrainte_borne_couches_internes,
    contrainte_quadratique_bornes,
    contrainte_ReLU_Mix,
    contrainte_derniere_couche_lineaire
)
from MOSEK_contraintes_generiques import (
    contrainte_boule_initiale,
    contrainte_premier_terme_egal_a_1
)
from MOSEK_coupes import(
    contrainte_McCormick_zk2,
    contrainte_Mc_Cormick_betai_betaj,
    coupes_RLT_LAN,
)
from MOSEK_contraintes_par_couches import(
    contrainte_recurrence_matrices_couches
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("path sys : ", sys.path)

from reseau_train import Reseau  # noqa: E402

inf = 10e5


def solveMix_SDP_par_couches(
    cert,
    coupes: Dict[str, bool], 
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
            
            numvar = 0  # Variables "indépendantes" -rien ici
            numcon = sum(cert.n[1:cert.K]) * 2 + 5 * cert.n[cert.K] + sum(cert.n[1:cert.K]) + cert.n[0] + cert.K +1 + sum(cert.n[1:cert.K]) + cert.n[0] + 3 * cert.n[cert.K] - 2

            # Ajout contrainte sur les zk^2
            if coupes["zk2"]:
                numcon += (2 * sum(cert.n[1:cert.K]) + 3 * cert.n[0])
            # Ajout contrainte sur les betas linearisation par Fortet
            if cert.n[cert.K] > 2 and coupes["betaibetaj"]:
                numcon += (3 * (int(cert.n[cert.K]) - 1) * (int(cert.n[cert.K]) - 2) // 2)
            # Ajout contrainte RLT Lan
            if coupes["RLTLan"]:
                numcon += (3 * sum(cert.n[k]*cert.n[k+1] for k in range(cert.K)) + sum(cert.n[1:cert.K]) 
                           + 2 * sum((cert.n[k])*(cert.n[k]-1)//2 for k in range(1,cert.K)))

            # Ajoute 'numcon' contraintes vides.
            # Les contraintes n'ont pas de bornes initialement.
            task.appendcons(numcon)

            # Ajout des variables semi-définies du problème : ici la matrice représentant les z et celle des betas
            for k in range(cert.K):
                task.appendbarvars([1 + cert.n[k] + cert.n[k+1]])
            task.appendbarvars([cert.n[cert.K]])
            # Ajout des variables "indépendantes" de la matrice sdp (ici 0 variable)
            task.appendvars(numvar)

            # ------------ FONCTION OBJECTIF ------------------------------------
            objective_function_diff_par_couches(task,cert.K,cert.n,cert.y0,numvar)
            # --------------------------------------------------------------------

            # ------------ CONTRAINTES RELU  ------------------------------------
            num_contrainte = 0

            # ***** Contrainte 2 :  zk+1 >= Wk zk + bk ********************
            # ***** Contrainte 3 :  zk+1 x (zk+1 - Wk zk - bk)  == 0  *****
            
            num_contrainte = contrainte_ReLU_Mix(task,cert.K,cert.n,cert.W,cert.b,num_contrainte, par_couches = True,
                                                 neurones_actifs_stables=cert.neurones_actifs_stables,
                                                 neurones_inactifs_stables=cert.neurones_inactifs_stables)

            # ***** Contrainte 4 :   zK+1 == WK zK + bK *****
            num_contrainte = contrainte_derniere_couche_lineaire(task,cert.K,cert.n,cert.W,cert.b,num_contrainte, par_couches = True)

            # ***** Contrainte 5 :  u (1 - betaj) + zKj > zKj*  *****  
            num_contrainte = contrainte_exemple_adverse_beta_u(task,cert.K,cert.n,cert.y0,cert.U,cert.rho,num_contrainte, par_couches = True)
            # ***** Contrainte 6 : somme(betaj) > 1 *****************
            num_contrainte = contrainte_exemple_adverse_somme_beta_superieure_1(
                task,cert.K,cert.n,cert.y0,num_contrainte, par_couches= True, betas_z_unis= False
            )
            # ***** Contrainte 7 :   betaj == 0 ou betaj ==1  *****
            num_contrainte = contrainte_beta_discret(task,cert.K,cert.n,cert.y0,num_contrainte, 
                                                     par_couches = True, betas_z_unis= False)

            # ***** Contrainte 8 :   Bornes sur les zkj hidden layers   *****
            #num_contrainte = contrainte_borne_couches_internes(task,K,n,U,num_contrainte,par_couches = True)
            num_contrainte = contrainte_quadratique_bornes(task,cert.K,cert.n,cert.L,cert.U,cert.x0,cert.epsilon,num_contrainte,par_couches=True)

            # ***** Contrainte 9 :   x - epsilon < z0 < x + epsilon  *****
            num_contrainte = contrainte_boule_initiale(task,cert.n,cert.x0,cert.epsilon,cert.U,cert.L,num_contrainte)

            # Contrainte 10 : X00 = 1 (Le premier terme de la matrice variable est 1)
            
            num_contrainte = contrainte_premier_terme_egal_a_1(task,cert.K,cert.K+1,num_contrainte)

            # Contrainte 11 : Assure la continuité entre les différents Pi
            # Pi+1[xi+1] = Pi[xi+1]
            num_contrainte = contrainte_recurrence_matrices_couches(task,cert.K,cert.n,num_contrainte)
            if verbose : 
                print("num contrainte fin des contraintes classiques : ", num_contrainte)
            # ***********************************************
            # ************ COUPES ***************************
            # ***********************************************
            # Contrainte 11 : Bornes sur zk^2 (RLT)
            if coupes["zk2"]:
                num_contrainte = contrainte_McCormick_zk2(task, cert.K, cert.n, cert.x0, cert.U, cert.L, cert.epsilon, num_contrainte, 
                                                          par_couches = True, neurones_actifs_stables=cert.neurones_actifs_stables,
                                                          neurones_inactifs_stables=cert.neurones_inactifs_stables)
                print("num contrainte apres zk^2 : ", num_contrainte)
            if coupes["betaibetaj"]:
                # Contrainte 12 : Linearisation de Fortet sur les betas
                num_contrainte = contrainte_Mc_Cormick_betai_betaj(task,cert.K, cert.n, cert.y0, num_contrainte, par_couches = True)
            if coupes["RLTLan"]:
                num_contrainte = coupes_RLT_LAN(task, cert.K, cert.n, cert.W, cert.b, cert.x0, cert.epsilon, cert.L, cert.U,num_contrainte, par_couches = True)
                print("num contrainte apres RLT Lan : ", num_contrainte)
            print("Nombre de contraintes : ", num_contrainte)
            # Configurer le solveur pour une optimisation
            task.putobjsense(mosek.objsense.minimize)

            # Write the problem for human inspection
            task.writedata("Models_MOSEK/ptf/Model_Mix_couches.ptf")

            # Résoudre le problème
            start_time = time.time()
            task.optimize()
            end_time = time.time()

            time_execution = end_time - start_time

            # Extraire la solution
            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.itr)

            status = -1
            Sol = []
            num_iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
            print("Mix couches Solsta : ", solsta)
            if solsta == solsta.optimal:
                # Assuming the optimization succeeded read solution
                z_sol = task.getbarxj(mosek.soltype.itr, 0)
                beta_sol = task.getbarxj(mosek.soltype.itr, cert.K)

                z_0 = reconstitue_matrice(1 + cert.n[0] + cert.n[1], z_sol)
                affiche_matrice(cert,z_0,"Mix_couches_SDP",titre,coupes,nom_variable="z_0")
                for i in range(1,cert.K):
                    z_sol_i = task.getbarxj(mosek.soltype.itr, i)
                    z_i = reconstitue_matrice(1 + cert.n[i] + cert.n[i+1], z_sol_i)
                    affiche_matrice(cert,z_i,"Mix_couches_SDP",titre,coupes,nom_variable=f"z_{i}")

                beta = reconstitue_matrice(cert.n[cert.K], beta_sol)
                affiche_matrice(cert,beta,"Mix_couches_SDP",titre,coupes,nom_variable="beta")

                if verbose : 
                    print("z : ", z_0)
                    print("beta: ", beta)

                # Obtenir la valeur du problème primal
                primal_obj_value = task.getprimalobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème primal: {primal_obj_value}")

                # Obtenir la valeur du problème dual
                dual_obj_value = task.getdualobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème dual: {dual_obj_value}")
                status = 1
                for j in range(cert.n[0]):
                    Sol.append(z_0[0][1+j])
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

            return Sol,  None, status, time_execution, {"Nombre_iterations" : num_iterations}



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
    coupes: Dict[str, bool] = {"zk^2": False, "betai*betaj": True, "RLT_Lan" : False

    }
    rho = 0.01
    epsilon = 5
    relax = 0
    ### Résoudre le problème QP
    solution, primal_obj_value, status, time_execution, dic_num_iterations = solveMix_SDP_par_couches(
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
