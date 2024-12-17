import mosek
import numpy as np
import time
import os
import sys
from typing import Dict

from MOSEK_objective import objective_function_diff_betas
from MOSEK_outils import (
    reconstitue_matrice,
    adapte_parametres_mosek,
    affiche_matrice
    )
from MOSEK_contraintes_adversariales import (
    contrainte_exemple_adverse_somme_beta_egale_1,
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
    contrainte_Mc_Cormick_betai_betaj_unis,
    contrainte_McCormick_zk_sigmak,
    coupes_RLT_LAN
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("path sys : ", sys.path)

#from reseau_train import Reseau  # noqa: E402

inf = 10e5


# MODELE PAS ENCORE COMPLET!!

def solveFprG_SDP_Adv2(
    cert, coupes : Dict, titre, verbose=True
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
                sum(cert.n[1:cert.K]) * 6 + 5 * cert.n[cert.K] + cert.n[0] - 2
            )
            # Ajout enveloppe de McCormick
            if coupes["zk^2"] :
                numcon+= (2 * sum(cert.n[1:cert.K]) + 3 * cert.n[0])
            # Ajout enveloppe sur les betas linearisation par Fortet
            if cert.n[cert.K] > 2 and coupes["betai*betaj"]:
                numcon += (3 * (int(cert.n[cert.K]) - 1) * (int(cert.n[cert.K]) - 2) // 2)
            # Ajout enveloppe de McCormick sur les zkj * sigmakj
            if coupes["sigmak*zk"] :
                numcon +=  (3 * sum(cert.n[1:cert.K]))
            if coupes["RLT_Lan"]:
                numcon += (3 * sum(cert.n[k]*cert.n[k+1] for k in range(cert.K)) + sum(cert.n[1:cert.K]) 
                           + 2 * sum((cert.n[k])*(cert.n[k]-1)//2 for k in range(1,cert.K)))

            
            if verbose :
                print("Nombre de contraintes totales : ", numcon)
            # Ajoute 'numcon' contraintes vides.
            # Les contraintes n'ont pas de bornes initialement.
            task.appendcons(numcon)

            # Ajout des variables SDP
            task.appendbarvars(
                [sum(cert.n) + sum(cert.n[1:cert.K]) + cert.n[cert.K]]
            )  # Matrice des z, des sigmas et des betas
            

            # Ajout des variables "indépendantes" de la matrice sdp (ici 0 variable)
            task.appendvars(numvar)

            # ------------ FONCTION OBJECTIF ------------------------------------
            objective_function_diff_betas(task,cert.K,cert.n,cert.y0,numvar,sigmas=True,par_couches=False)
            # --------------------------------------------------------------------

            # ------------ CONTRAINTES RELU  ------------------------------------
            num_contrainte = 0

            
            # ***** Contrainte 1,2 : l (1-sigma) <= Wz+b <= u sigma *****
            # ***** Contrainte 3 : sigmaK x (WK zK + bK) = zK+1 *****
            num_contrainte = contrainte_ReLU_Glover(
                task, cert.K, cert.n, cert.W, cert.b, cert.U, cert.L, num_contrainte
            )
            num_contrainte = contrainte_sigma_borne(task,cert.K,cert.n,num_contrainte)
            if verbose : 
                print("Nombre de contraintes actuelles apres 1,2,3 : ", num_contrainte)

            
            # ***** Contrainte 4 : La derniere couche est juste lineaire zK+1 = WK zK + bK *****
            num_contrainte = contrainte_derniere_couche_lineaire(
                task, cert.K, cert.n, cert.W, cert.b, num_contrainte
            )
            if verbose : 
                print("Nombre de contraintes actuelles apres 4 : ", num_contrainte)

            # ***** Contrainte 5 : somme(betaj) = 1 *****************
            num_contrainte = contrainte_exemple_adverse_somme_beta_egale_1(
                task,cert.K,cert.n,cert.y0,cert.U,cert.rho,num_contrainte, par_couches = False, betas_z_unis= True
            )
            
            if verbose : 
                print("Nombre de contraintes actuelles apres 5 : ", num_contrainte)

            # ***** Contrainte 6 : les betaj sont discrets  *****
            num_contrainte = contrainte_beta_discret(task, cert.K, cert.n, cert.y0, num_contrainte, 
                                                     par_couches = False, betas_z_unis= True)
            if verbose :
                print("Nombre de contraintes actuelles apres 6 : ", num_contrainte)

            # ***** Contrainte 7 :  0 <= betaj <= 1 ***************************
            num_contrainte = contrainte_borne_betas(task,cert.K,cert.n,cert.y0,cert.U,cert.L,cert.rho,num_contrainte,
                                                    sigmas = True, par_couches = False, betas_z_unis= True)
            if verbose : 
                print("Nombre de contraintes après contrainte 7 : ", num_contrainte)

            # ***** Contrainte 8 : 0 <= betaj zK_j <= U betaj, 0 <= betaj zK_j <= zK_j
            num_contrainte = contrainte_borne_betas_unis(task,cert.K,cert.n,cert.y0,cert.U,cert.L,cert.rho,num_contrainte,
                                                         sigmas = True, par_couches= False)
            if verbose : 
                print("Nombre de contraintes après contrainte 8 : ", num_contrainte)

            # ***** Contrainte 9 :   x - epsilon < z0 < x + epsilon  *****
            num_contrainte = contrainte_boule_initiale(
                task, cert.n, cert.x0, cert.epsilon, cert.U, cert.L, num_contrainte
            )
            if verbose : 
                print("Nombre de contraintes actuelles apres 9 : ", num_contrainte)

            
            # ***** Contrainte 10 : Bornes sur les couches internes zk *****
            num_contrainte = contrainte_borne_couches_internes(
                task, cert.K, cert.n, cert.U, num_contrainte
            )
            if verbose :
                print("Nombre de contraintes actuelles apres 10 : ", num_contrainte)

            # ***** Contrainte 11 : *****
            num_contrainte = contrainte_sigma_discret(task, cert.K, cert.n, num_contrainte)
            if verbose : 
                print("Nombre de contraintes actuelles apres 11 : ", num_contrainte)

            # Contrainte X00 = 1
            num_contrainte = contrainte_premier_terme_egal_a_1(task,cert.K,1,num_contrainte)
            print("Nombre de contraintes actuelles apres X00=1 : ", num_contrainte)


            # Contrainte 11 : Enveloppes de McCormick sur zk^2
            if coupes["zk^2"] : 
                num_contrainte = contrainte_McCormick_zk2(
                    task, cert.K, cert.n, cert.x0, cert.U, cert.epsilon, num_contrainte
                )
                #if verbose : 
                print("Nombre de contraintes actuelles apres 11 : ", num_contrainte)

            # Contrainte 12 : Linearisation de Fortet sur les betas
            if coupes["betai*betaj"] :
                num_contrainte = contrainte_Mc_Cormick_betai_betaj_unis(
                    task, cert.K, cert.n, cert.y0, num_contrainte, sigmas = True, par_couches= False
                )
                #if verbose :
                print("Nombre de contraintes actuelles apres 12 : ", num_contrainte)

            # Contrainte 13 : Enveloppes de McCormick sur le produit zkj * sigmakj
            if coupes["sigmak*zk"] : 
                num_contrainte = contrainte_McCormick_zk_sigmak(
                    task, cert.K, cert.n, cert.U, num_contrainte
                )
                #if verbose : 
                print("Nombre de contraintes actuelles apres 13 : ", num_contrainte)
            
            if coupes["RLT_Lan"]:
                num_contrainte = coupes_RLT_LAN(task,cert.K,cert.n,cert.W,cert.b,cert.x0,cert.epsilon,cert.L,cert.U,num_contrainte)
                print("Nombre de contraintes apres RLT LAN : ", num_contrainte)
            print("Nombre de contraintes : ", num_contrainte)
            # Configurer le solveur pour une optimisation
            task.putobjsense(mosek.objsense.minimize)

            task.writedata("Models_MOSEK/ptf/Model_FprG_d.ptf")

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
                z = reconstitue_matrice(sum(cert.n) + sum(cert.n[1:cert.K]) + cert.n[cert.K], z_sol)
                affiche_matrice(cert, z,"FprG_d", titre)

                # Obtenir la valeur du problème primal
                primal_obj_value = task.getprimalobj(mosek.soltype.itr)
                if verbose :
                    print(f"Valeur du problème primal: {primal_obj_value}")

                # Obtenir la valeur du problème dual
                dual_obj_value = task.getdualobj(mosek.soltype.itr)
                if verbose : 
                    print(f"Valeur du problème dual: {dual_obj_value}")

                Sol = []
                for j in range(cert.n[0]):
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

            # Obtenir le diagnostic de reparation du modele
            num_repair = numcon  
            wlc = [1.0] * num_repair
            wuc = [1.0] * num_repair
            wlx = [1.0] * num_repair
            wux = [1.0] * num_repair

            task.primalrepair(wlc, wuc, wlx, wux)

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
    y0= 1
    x0 = [0.6, -0.3, 0.4]
    #y0 = Res.retourne_label(x0)
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
    solution, primal_obj_value, status, time_execution, dic_num_iterations = solveFprG_SDP_Adv2(
        K, n, x0, y0, ycible, U, L, W_reverse, b, epsilon, rho, relax, coupes, verbose=1
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
