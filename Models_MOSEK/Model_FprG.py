import mosek
import numpy as np
import time
import os
import sys
from typing import Dict, List

from MOSEK_objective import objective_function_diff
from MOSEK_outils import (
    reconstitue_matrice,
    adapte_parametres_mosek
    )
from MOSEK_contraintes_adversariales import (
    contrainte_exemple_adverse_beta_u,
    contrainte_exemple_adverse_somme_beta_superieure_1,
    contrainte_beta_discret,
    contrainte_borne_betas,
    contrainte_borne_betas_unis
)
from MOSEK_contraintes_passage_couches import (
    contrainte_borne_couches_internes,
    contrainte_ReLU_Glover,
    contrainte_sigma_borne,
    contrainte_sigma_discret,
    contrainte_derniere_couche_lineaire,
)
from MOSEK_contraintes_generiques import (
    contrainte_boule_initiale,
    contrainte_premier_terme_egal_a_1,
)
from MOSEK_coupes import (
    contrainte_McCormick_zk2,
    contrainte_Mc_Cormick_betai_betaj,
    contrainte_McCormick_zk_sigmak,
    coupes_RLT_LAN
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("path sys : ", sys.path)

from reseau_train import Reseau  # noqa: E402

inf = 10e5




def solveFprG_SDP(
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
    rho : float, 
    coupes: Dict[str, bool] = {"zk^2": True, "betai*betaj": True, "sigmak*zk" : True, "RLT_Lan" : False}, verbose=1
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
            numcon = (
                sum(n[1:K]) * 6 + 3 * n[K] + n[0] + 1
            )
            # Ajout enveloppe de McCormick
            if coupes["zk^2"] :
                numcon+= (2 * sum(n[1:K]) + 3 * n[0])
            # Ajout enveloppe sur les betas linearisation par Fortet
            if n[K] > 2 and coupes["betai*betaj"]:
                numcon += (3 * (int(n[K]) - 1) * (int(n[K]) - 2) // 2)
            # Ajout enveloppe de McCormick sur les zkj * sigmakj
            if coupes["sigmak*zk"] :
                numcon +=  (3 * sum(n[1:K]))
            if coupes["RLT_Lan"]:
                numcon += (3 * sum(n[k]*n[k+1] for k in range(K)) + sum(n[1:K]) 
                           + 2 * sum((n[k])*(n[k]-1)//2 for k in range(1,K)))

            
            if verbose :
                print("Nombre de contraintes totales : ", numcon)
            # Ajoute 'numcon' contraintes vides.
            # Les contraintes n'ont pas de bornes initialement.
            task.appendcons(numcon)

            # Ajout des variables SDP
            task.appendbarvars(
                [sum(n) + sum(n[1:K]) + 1]
            )  # Matrice des z et des sigmas
            task.appendbarvars([n[K]])  # Matrice des betas

            # Ajout des variables "indépendantes" de la matrice sdp (ici 0 variable)
            task.appendvars(numvar)

            # ------------ FONCTION OBJECTIF ------------------------------------
            objective_function_diff(task, K, n, ytrue, numvar)
            # --------------------------------------------------------------------

            # ------------ CONTRAINTES RELU  ------------------------------------
            num_contrainte = 0

            
            # ***** Contrainte 1,2 : l (1-sigma) <= Wz+b <= u sigma *****
            # ***** Contrainte 3 : sigmaK x (WK zK + bK) = zK+1 *****
            num_contrainte = contrainte_ReLU_Glover(
                task, K, n, W, b, U, L, num_contrainte
            )
            if verbose : 
                print("Nombre de contraintes actuelles apres 1,2,3 : ", num_contrainte)

            num_contrainte = contrainte_sigma_borne(task,K,n,num_contrainte)
            # ***** Contrainte 4 : La derniere couche est juste lineaire zK+1 = WK zK + bK *****
            num_contrainte = contrainte_derniere_couche_lineaire(
                task, K, n, W, b, num_contrainte
            )
            if verbose : 
                print("Nombre de contraintes actuelles apres 4 : ", num_contrainte)

            # ***** Contrainte 5 :  u (1 - betaj) + zKj > zKj*  *****  
            num_contrainte = contrainte_exemple_adverse_beta_u(
                task, K, n, ytrue, U, rho, num_contrainte
            )
            # ***** Contrainte 6 : somme(betaj) > 1 *****************
            num_contrainte = contrainte_exemple_adverse_somme_beta_superieure_1(
                task,K,n,ytrue,U,rho,num_contrainte, par_couches = False, betas_z_unis= False
            )
            
            if verbose : 
                print("Nombre de contraintes actuelles apres 5,6 : ", num_contrainte)

            # ***** Contrainte 7 : les betaj sont discrets  *****
            num_contrainte = contrainte_beta_discret(task, K, n, ytrue, num_contrainte, 
                                                     par_couches = False, betas_z_unis= False)
            if verbose :
                print("Nombre de contraintes actuelles apres 7 : ", num_contrainte)


            # ***** Contrainte 8 :   x - epsilon < z0 < x + epsilon  *****
            num_contrainte = contrainte_boule_initiale(
                task,n,x0,epsilon,U,L,num_contrainte
                )
            if verbose : 
                print("Nombre de contraintes actuelles apres 8 : ", num_contrainte)

            
            # ***** Contrainte 9 : Bornes sur les couches internes zk *****
            num_contrainte = contrainte_borne_couches_internes(
                task, K, n, U, num_contrainte
            )
            if verbose :
                print("Nombre de contraintes actuelles apres 9 : ", num_contrainte)

            # ***** Contrainte 10 : *****
            num_contrainte = contrainte_sigma_discret(task, K, n, num_contrainte)
            if verbose : 
                print("Nombre de contraintes actuelles apres 10 : ", num_contrainte)

            # Contrainte X00 = 1
            num_contrainte = contrainte_premier_terme_egal_a_1(task,K,2,num_contrainte)
            #if verbose : 
            print("Nombre de contraintes actuelles apres X00=1 : ", num_contrainte)


            # Contrainte 11 : Enveloppes de McCormick sur zk^2
            if coupes["zk^2"] : 
                num_contrainte = contrainte_McCormick_zk2(
                    task, K, n, x0, U, epsilon, num_contrainte
                )
                #if verbose : 
                print("Nombre de contraintes actuelles apres 11 : ", num_contrainte)

            # Contrainte 12 : Linearisation de Fortet sur les betas
            if coupes["betai*betaj"] :
                num_contrainte = contrainte_Mc_Cormick_betai_betaj(
                    task, K, n, ytrue, num_contrainte
                )
                #if verbose :
                print("Nombre de contraintes actuelles apres 12 : ", num_contrainte)

            # Contrainte 13 : Enveloppes de McCormick sur le produit zkj * sigmakj
            if coupes["sigmak*zk"] : 
                num_contrainte = contrainte_McCormick_zk_sigmak(
                    task, K, n, U, num_contrainte
                )
                #if verbose : 
                print("Nombre de contraintes actuelles apres 13 : ", num_contrainte)
            
            if coupes["RLT_Lan"]:
                num_contrainte = coupes_RLT_LAN(task,K,n,W,b,x0,epsilon,L,U,num_contrainte)
                print("Nombre de contraintes apres RLT LAN : ", num_contrainte)
            print("Nombre de contraintes : ", num_contrainte)
            # Configurer le solveur pour une optimisation
            task.putobjsense(mosek.objsense.minimize)

            # Write the problem for human inspection

            task.writedata("Models_MOSEK/ptf/Model_FprG.ptf")

            # Résoudre le problème
            start_time = time.time()
            task.optimize()
            end_time = time.time()

            time_execution = end_time - start_time

            # Extraire la solution
            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.itr)
            status = -1
            num_iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
            if solsta == solsta.optimal:
                # Assuming the optimization succeeded read solution

                z_sol = task.getbarxj(mosek.soltype.itr, 0)
                beta_sol = task.getbarxj(mosek.soltype.itr, 1)

                z = reconstitue_matrice(sum(n) + sum(n[1:K]) + 1, z_sol)
                beta = reconstitue_matrice(n[K], beta_sol)

                if verbose:
                    print(
                        "beta : ",
                        [
                            [round(beta[i][j], 3) for j in range(n[K])]
                            for i in range(n[K])
                        ],
                    )
                    print(
                        "z : ",
                        [
                            [round(z[i][j], 2) for j in range(sum(n) + sum(n[1:K]) + 1)]
                            for i in range(sum(n) + sum(n[1:K]) + 1)
                        ],
                    )

                # Obtenir la valeur du problème primal
                primal_obj_value = task.getprimalobj(mosek.soltype.itr)
                if verbose :
                    print(f"Valeur du problème primal: {primal_obj_value}")

                # Obtenir la valeur du problème dual
                dual_obj_value = task.getdualobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème dual: {dual_obj_value}")

                Sol = []
                for j in range(n[0]):
                    Sol.append(z[0, j + 1])

                status = 1
                return Sol, primal_obj_value,status,  time_execution, {"Nombre_iterations" : num_iterations}

            elif solsta == solsta.dual_infeas_cer or solsta == solsta.prim_infeas_cer:
                if verbose: 
                    print("Primal or dual infeasibility certificate found.\n")
                status = 3
            elif solsta == solsta.unknown:
                if verbose :
                    print("Unknown solution status")
            else:
                if verbose : 
                    print("Other solution status")

            # Obtain problem diagnostic information
            # Setting up repair parameters
            # num_repair = numcon  # Adjust as necessary for your problem size
            # wlc = [1.0] * num_repair
            # wuc = [1.0] * num_repair
            # wlx = [1.0] * num_repair
            # wux = [1.0] * num_repair

            # # Perform repair
            # task.primalrepair(wlc, wuc, wlx, wux)

            return [], None,status, time_execution, {"Nombre_iterations" : num_iterations}

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
    print("W reverse : ", W_reverse)
    b = [[-0.2, 0.1], [0.3, -0.2, 0.3, -0.1], [0.2, -0.5, 0.1]]
    n = [3, 2, 4, 3]

    u = 150
    lb = -15

    #Res = Reseau(K, n, W_reverse, b)

    x0 = [0.6, -0.3, 0.4]
    y0 = Res.retourne_label(x0)
    print("Label x0 : ", y0)

    ycible = 2
    U = [[u] * taille for taille in n]
    L = [[lb] * taille for taille in n]
    print("U : ", U)

    rho = 0.01
    epsilon = 5
    relax = 0
    coupes: Dict[str, bool] = {"zk^2": True, "betai*betaj": True, "sigmak*zk" : True, "RLT_Lan" : True}
    ### Résoudre le problème QP
    solution, primal_obj_value, status, time_execution, dic_num_iterations = solveFprG_SDP(
        K, n, x0, y0, ycible, U, L, W_reverse, b, epsilon, rho, coupes, verbose=1
    )
    print("Solution:", solution)
    print("Nombre d'itérations : ", dic_num_iterations["Nombre_iterations"])
    # if solution is not None:
    #     sortie = Res(solution, True)
    #     print("Sortie du réseau : ", sortie)
    #     predicted_label = Res.retourne_label(solution)
    #     print("Label : ", predicted_label)
    #     #print("Sortie en 0 : ", Res([0, 0, 0]))


if __name__ == "__main__":
    test()
