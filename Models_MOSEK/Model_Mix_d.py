import mosek
import numpy as np
import time
import os
import sys
from typing import Dict, List

from MOSEK_objective import (
    objective_function_diff_betas
)
from MOSEK_outils import(
    reconstitue_matrice,
    adapte_parametres_mosek,
    affiche_matrice,
    save_matrice, 
    tableau_matrice_csv,
    imprime_ptf
)
from MOSEK_contraintes_adversariales import(
    contrainte_exemple_adverse_somme_beta_egale_1,
    contrainte_exemple_adverse_beta_produit_complet,
    contrainte_exemple_adverse_beta_produit_simple,
    contrainte_beta_discret,
    contrainte_borne_betas,
    contrainte_borne_betas_unis,
    contrainte_produit_betas_nuls_Adv2_Adv3,
    contrainte_borne_somme_betaz,
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
    contrainte_Mc_Cormick_betai_betaj_unis,
    coupes_RLT_LAN,
    contrainte_Mc_Cormick_betai_zkj
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("path sys : ", sys.path)

from reseau_train import Reseau  # noqa: E402

inf = 10e5





def solveMix_SDP_objbetas(
    cert,
    coupes: Dict[str, bool], 
    titre : str,
    derniere_couche_lineaire : bool = True,
    verbose : bool =True
):
    def streamprinter(text):
        if verbose : 
            sys.stdout.write(text)
            sys.stdout.flush()
        else : 
            pass  # N'affiche aucun log du solveur
    with mosek.Env() as env:
        with env.Task() as task:
            
            task.set_Stream(mosek.streamtype.log, streamprinter)
            adapte_parametres_mosek(task)
            numvar = 0  # Variables "indépendantes" -rien ici
            numcon = sum(cert.n[1:cert.K]) * 3 + 5 * cert.n[cert.K] + sum(cert.n[1:cert.K]) + cert.n[0] + (cert.n[cert.K]-1) * 3 + cert.n[0] - 1 + 3 * cert.n[cert.K]
            # Ajout contrainte sur les zk^2
            if coupes["zk2"]:
                numcon += (2 * sum(cert.n[1:cert.K]) + 3 * cert.n[0])
            # Ajout contrainte sur les betas linearisation par Fortet
            if cert.n[cert.K] > 2 and coupes["betaibetaj"]:
                # numcon += (3 * (int(n[K]) - 1) * (int(n[K]) - 2) // 2)
                numcon += int(cert.n[cert.K] * (cert.n[cert.K]-1) // 2)
            # Ajout contrainte RLT Lan
            if coupes["RLTLan"]:
                if verbose : 
                    print("AJOUT CONTRAINTES RLT")
                numcon += ( 3 * sum(cert.n[k]*cert.n[k+1] for k in range(cert.K)) + sum(cert.n[1:cert.K]) 
                           + 2 * sum((cert.n[k])*(cert.n[k]-1)//2 for k in range(1,cert.K)) )

            if coupes["betaizkj"]:
                numcon += (2 * (cert.n[cert.K]-1) * sum(cert.n))  # coupes betai * zkj

            if coupes["bornes_betaz"]:
                numcon += 1
            #print("numcon : ", numcon)
            task.appendcons(numcon)

            # Ajout des variables semi-définies du problème : ici la matrice représentant les z et celle des betas
            if derniere_couche_lineaire :
                task.appendbarvars([sum(cert.n) + cert.n[cert.K]])
            else : 
                task.appendbarvars([sum(cert.n[:cert.K]) + cert.n[cert.K]])
            # Ajout des variables "indépendantes" de la matrice sdp (ici 0 variable)
            task.appendvars(numvar)

            # ------------ FONCTION OBJECTIF ------------------------------------
            objective_function_diff_betas(task,cert.K,cert.n,cert.W,cert.y0,numvar,derniere_couche_lineaire=derniere_couche_lineaire)
            # --------------------------------------------------------------------

            # ------------ CONTRAINTES RELU  ------------------------------------
            num_contrainte = 0

            # ***** Contrainte 1 :  zk+1 >= Wk zk + bk ********************
            # ***** Contrainte 2 :  zk+1 x (zk+1 - Wk zk - bk)  == 0  *****
            num_contrainte = contrainte_ReLU_Mix(task,cert.K,cert.n,cert.W,cert.b,num_contrainte,
                                                 neurones_actifs_stables=cert.neurones_actifs_stables,
                                                 neurones_inactifs_stables=cert.neurones_inactifs_stables)
            if verbose : 
                print("Nombre de contraintes après contrainte 1-2", num_contrainte)

            # ***** Contrainte 3 :   zK+1 == WK zK + bK *******************
            if derniere_couche_lineaire :
                num_contrainte = contrainte_derniere_couche_lineaire(task,cert.K,cert.n,cert.W,cert.b,num_contrainte)
            if verbose : 
                print("Nombre de contraintes après contrainte 3", num_contrainte)

            # ***** Contrainte 4 : somme(betaj) = 1 ***********************
            num_contrainte = contrainte_exemple_adverse_somme_beta_egale_1(task,cert.K,cert.n,cert.y0,num_contrainte, 
                                                                           par_couches=False, betas_z_unis=True, derniere_couche_lineaire=derniere_couche_lineaire)
            if verbose : 
                print("Nombre de contraintes après contrainte 4", num_contrainte)

            # ***** Contrainte 5 : betaj zj >= betaj zjtrue **************
            num_contrainte = contrainte_exemple_adverse_beta_produit_simple(task,cert.K,cert.n,cert.W,cert.y0,cert.U,cert.rho,num_contrainte,
                                                                            derniere_couche_lineaire=derniere_couche_lineaire)
            if verbose : 
                print("Nombre de contraintes après contrainte 5", num_contrainte)

            # ***** Contrainte 6 :   betaj == 0 ou betaj ==1  *************
            num_contrainte = contrainte_beta_discret(task,cert.K,cert.n,cert.y0,num_contrainte, 
                                                     par_couches = False, betas_z_unis = True, derniere_couche_lineaire=derniere_couche_lineaire)
            if verbose: 
                print("Nombre de contraintes après contrainte 6", num_contrainte)

            # ***** Contrainte 7 :  0 <= betaj <= 1 ***************************
            num_contrainte = contrainte_borne_betas(task,cert.K,cert.n,cert.y0,num_contrainte,
                                                    par_couches = False, betas_z_unis= True, derniere_couche_lineaire=derniere_couche_lineaire)
            if verbose : 
                print("Nombre de contraintes après contrainte 7", num_contrainte)

            # ***** Contrainte 7bis : 0 <= betaj zK_j <= U betaj, 0 <= betaj zK_j <= zK_j
            num_contrainte = contrainte_borne_betas_unis(task,cert.K,cert.n,cert.W,cert.y0,cert.U,cert.L,cert.rho,num_contrainte,
                                                         par_couches= False,derniere_couche_lineaire=derniere_couche_lineaire)
            if verbose : 
                print("Nombre de contraintes après contrainte 7bis", num_contrainte)
            

            # ***** Contrainte 8 :   Bornes sur les zkj hidden layers   *****
            # num_contrainte = contrainte_borne_couches_internes(task,K,n,U,num_contrainte)
            num_contrainte = contrainte_quadratique_bornes(task,cert.K,cert.n,cert.L,cert.U,cert.x0,cert.epsilon,num_contrainte)
            num_contrainte = contrainte_borne_couches_internes(task, cert.K, cert.n, cert.U, num_contrainte, par_couches = False)
            if verbose : 
                print("Nombre de contraintes après contrainte 8", num_contrainte)

            # ***** Contrainte 9 :   x - epsilon < z0 < x + epsilon  *****
            num_contrainte = contrainte_boule_initiale(task,cert.n,cert.x0,cert.epsilon,cert.U,cert.L,num_contrainte)
            if verbose : 
                print("Nombre de contraintes après contrainte 9", num_contrainte)

            # Contrainte : somme(betaj zKj) <= max(U[K]) **************************
            num_contrainte = contrainte_borne_somme_betaz(task,cert.K,cert.n,cert.y0,cert.U,num_contrainte,par_couches=False,derniere_couche_lineaire=derniere_couche_lineaire)
           
            # Contrainte 10 : X00 = 1 (Le premier terme de la matrice variable est 1)
            num_contrainte = contrainte_premier_terme_egal_a_1(task,cert.K,1,num_contrainte)
            if verbose : 
                print("Nombre de contraintes après contrainte 10", num_contrainte)
            # ***********************************************
            # ************ COUPES ***************************
            # ***********************************************
            # Contrainte 11 : Bornes sur zk^2 (RLT)
            if coupes["zk2"]:
                num_contrainte = contrainte_McCormick_zk2(task, cert.K, cert.n, cert.x0, cert.U, cert.L, cert.epsilon, num_contrainte,
                                                          par_couches=False, neurones_actifs_stables=cert.neurones_actifs_stables,
                                                          neurones_inactifs_stables=cert.neurones_inactifs_stables)

            if coupes["betaibetaj"]:
                # Contrainte 12 : Linearisation de Fortet sur les betas
                # num_contrainte = contrainte_Mc_Cormick_betai_betaj_unis(task,K,n,ytrue,num_contrainte)
                num_contrainte = contrainte_produit_betas_nuls_Adv2_Adv3(task,cert.K,cert.n,cert.y0,num_contrainte)

            if coupes["RLTLan"]:
                num_contrainte = coupes_RLT_LAN(task,cert.K,cert.n,cert.W,cert.b,cert.x0,cert.epsilon,cert.L,cert.U,num_contrainte)

            if coupes["betaizkj"]:
                num_contrainte = contrainte_Mc_Cormick_betai_zkj(task, cert.K, cert.n, cert.y0, cert.U, num_contrainte)
            
            if coupes["bornes_betaz"]:
                num_contrainte = contrainte_borne_somme_betaz(task,cert.K,cert.n,cert.y0,cert.U,num_contrainte,par_couches=False)
                                                            
                                                            
            #print("Nombre de contraintes après contrainte McCormick betai zkj", num_contrainte)
            if verbose : 
                print("Nombre final de contraintes : ", num_contrainte)

            # Configurer le solveur pour une optimisation
            task.putobjsense(mosek.objsense.minimize)
            task.writedata(os.path.join(os.getcwd(),"Models_MOSEK/ptf/Model_Mix_d.ptf"))
            imprime_ptf(cert,task,"Mix_d_SDP",titre,coupes)

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
            print(f"Mix d status: {solsta}")
            if solsta == solsta.optimal:
                z_sol = task.getbarxj(mosek.soltype.itr, 0)

                z = reconstitue_matrice(sum(cert.n) + cert.n[cert.K], z_sol)

                if cert.data_modele != "MNIST":
                    affiche_matrice(cert,z,"Mix_d_SDP",titre,coupes,nom_variable="zbeta")
                    tableau_matrice_csv(cert,z,"Mix_d_SDP",titre,coupes,nom_variable="zbeta")
                else :
                    n_rows, ncols = z.shape
                    mask_beta = np.zeros((n_rows), dtype = bool) 
                    mask_beta[(cert.n[0]) :( sum(cert.n) + cert.n[cert.K]) - 1] = True
                    mask_beta[0] = True
                    print("T mask sans reshape shape : ", z[np.outer(mask_beta, mask_beta)].shape)
                    T = z[np.outer(mask_beta, mask_beta)].reshape(sum(cert.n[1:]) + cert.n[cert.K], sum(cert.n[1:]) + cert.n[cert.K])
                    print("T mask shape : ", T.shape)
                    save_matrice(cert,T,
                                 "Mix_d_SDP", titre, coupes, nom_variable = "zbeta_sans_input")
                    tableau_matrice_csv(cert,z,"Mix_d_SDP",titre,coupes,nom_variable="zbeta")
                # print(
                #     "z : ",
                #     [
                #         [round(z[i][j], 2) for j in range(sum(cert.n) + 1)]
                #         for i in range(sum(cert.n) + 1)
                #     ],
                # )

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
                return Sol,  primal_obj_value, status, time_execution, {"Nombre_iterations" : num_iterations}
            elif solsta == solsta.dual_infeas_cer or solsta == solsta.prim_infeas_cer:
                if verbose :
                    print("Primal or dual infeasibility certificate found.\n")
                
                status = 3
            elif solsta == solsta.unknown:
                if verbose :
                    print("Unknown solution status")
                try : 
                    z_sol = task.getbarxj(mosek.soltype.itr, 0)   
                    z = reconstitue_matrice(sum(cert.n) + cert.n[cert.K], z_sol)
                    if verbose:
                        print(
                            "z : ",
                            [
                                [round(z[i][j], 2) for j in range(sum(cert.n) + 1)]
                                for i in range(sum(cert.n) + 1)
                            ],
                        )
                    for j in range(cert.n[0]):
                        Sol.append(z_sol[j + 1])
                    status = 2
                except Exception as e:
                    pass
            else:
                if verbose :
                    print("Other solution status")

            return Sol, None, status, time_execution, {"Nombre_iterations" : num_iterations}


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

    u = 150
    lb = -15

    Res = Reseau(K, n, W_reverse, b)

    x0 = [0.6, -0.3, 0.4]
    y0 = Res.retourne_label(x0)
    print("y0 : ", y0)

    ycible = 1
    U = [[u] * taille for taille in n]
    L = [[lb] * taille for taille in n]
    coupes: Dict[str, bool] = {"zk^2": False, "betai*betaj": False, "RLT_Lan" : False}
    rho = 0.01
    epsilon = 0.1
    relax = 0
    ### Résoudre le problème QP
    solution, status, primal_obj_value, time_execution, dic_num_iterations = solveMix_SDP_objbetas(
        K, n, x0, y0, ycible, U, L, W_reverse, b, epsilon, rho, coupes, verbose=1
    )
    print("Solution:", solution)
    print("Nombre d'itérations : ", dic_num_iterations["Nombre_iterations"])
    if solution is not None and solution != []:
        sortie = Res(solution, True)
        print("Sortie du réseau pour la solution calculée: ", sortie)
        predicted_label = Res.retourne_label(solution)
        print("Label retourné: ", predicted_label)

if __name__ == "__main__":
    test()
