import numpy as np
import mosek
from MOSEK_outils import return_good_j_beta, return_i_from_k_j__variable_z, return_i_from_k_j__variable_sigma, return_k_j_from_i
from typing import List, Tuple
inf = 10e5


# *******************************************************************    
# ******************* COUPES MC CORMICK *****************************
# *******************************************************************
def contrainte_McCormick_zk2(task : mosek.Task,
                            K : int, 
                            n : List[int], 
                            x0 : List[float], 
                            U : List[List[float]], 
                            L : List[List[float]], 
                            epsilon : float,
                            num_contrainte : int,
                            par_couches : bool = False,
                            neurones_actifs_stables : List = [],
                            neurones_inactifs_stables : List = [],
                            derniere_couche_lineaire : bool = True):
    """Genere une coupe de McCormick sur les zk^2 diagonaux"""
    # ***** Nombre de contraintes : sum(n[0:K]) *****************
    # Contrainte sur l'input
    for j in range(n[0]):
        ub = x0[j] + epsilon
        lb = x0[j] - epsilon
        
        # Contrainte : z0^2 >= 2(x0 - epsilon) z0 - (x0 - epsilon)^2 ***********
        task.putbarablocktriplet(
            [num_contrainte, num_contrainte],
            [0, 0],
            [j + 1, j + 1],
            [j + 1, 0],
            [1, -lb],
        )
        task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [-lb * lb], [inf])
        num_contrainte += 1

        # Contrainte : z0^2 >= 2(x0 + epsilon) z0 - (x0 + epsilon)^2
        task.putbarablocktriplet(
            [num_contrainte, num_contrainte],
            [0, 0],
            [j + 1, j + 1],
            [j + 1, 0],
            [1, -ub],
        )
        task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [-ub * ub], [inf])
        num_contrainte += 1

        # z0^2 <= 2 x0 z0 - (x0 + epsilon) (x0 - epsilon)
        task.putbarablocktriplet(
            [num_contrainte] * 2, [0] * 2, [j + 1, j + 1], [j + 1, 0], [1, -x0[j]]
        )
        task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [-ub * lb])
        num_contrainte += 1
    # Contrainte sur les hidden layers
    for k in range(1, K):
        for j in range(n[k]):
            if (k, j) in neurones_inactifs_stables + neurones_actifs_stables:
                continue

            #  Contrainte : zk^2  >= - U^2 + 2 U zk *********
            if par_couches:
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [k-1, k-1],
                    [1 + n[k-1] + j, 1 + n[k-1] + j],
                    [1  + n[k-1] + j, 0],
                    [1, - U[k][j]],
                )
            else :
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [0, 0],
                    [return_i_from_k_j__variable_z(k,j,n), return_i_from_k_j__variable_z(k,j,n)],
                    [return_i_from_k_j__variable_z(k,j,n), 0],
                    [1, -U[k][j]],
                )
            # Bornes
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.lo], [-U[k][j] * U[k][j]], [inf]
            )
            num_contrainte += 1

            # Contrainte : zk^2 <= U zk **********************
            if par_couches : 
                task.putbarablocktriplet(
                    [num_contrainte] * 2,
                    [k-1] * 2,
                    [1 + n[k-1] + j, 1 + n[k-1] + j],
                    [1 + n[k-1] + j, 0],
                    [1, -U[k][j] / 2],
                )
            else :
                task.putbarablocktriplet(
                    [num_contrainte] * 2,
                    [0] * 2,
                    [return_i_from_k_j__variable_z(k,j,n), return_i_from_k_j__variable_z(k,j,n)],
                    [return_i_from_k_j__variable_z(k,j,n), 0],
                    [1, -U[k][j] / 2],
                )
            # Bornes
            if U[k][j] * U[k][j] < 1e6:
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.ra], [-U[k][j] * U[k][j]], [0]
                )
            else:
                task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [0])
            num_contrainte += 1
    # Contrainte sur la derniere couche (pas forcement positive)
    if not derniere_couche_lineaire:
        return num_contrainte
    for j in range(n[K]):
        # ***** Contrainte U^2 - 2 zK U + zK^2 >= 0 *************************
        if par_couches:
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [K-1, K-1],
                [1 + n[K-1] + j, 1 + n[K-1] + j],
                [1  + n[K-1] + j, 0],
                [1, - U[K][j]],
            )
        else :
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [0, 0],
                [return_i_from_k_j__variable_z(K,j,n), return_i_from_k_j__variable_z(K,j,n)],
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1, -U[K][j]],
            )
        # Bornes
        task.putconboundlist(
            [num_contrainte], [mosek.boundkey.lo], [-U[K][j] * U[K][j]], [inf]
        )
        num_contrainte += 1


        # ***** Contrainte L^2 - 2 zK L + zK^2 >= 0 *************************
        if par_couches:
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [K-1, K-1],
                [1 + n[K-1] + j, 1 + n[K-1] + j],
                [1  + n[K-1] + j, 0],
                [1, - L[K][j]],
            )
        else :
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [0, 0],
                [return_i_from_k_j__variable_z(K,j,n), return_i_from_k_j__variable_z(K,j,n)],
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1, -L[K][j]],
            )
        # Bornes
        task.putconboundlist(
            [num_contrainte], [mosek.boundkey.lo], [-L[K][j] * L[K][j]], [inf]
        )
        num_contrainte += 1

        # ***** Contrainte zK^2 <= zK U + L zK - LU *************************
        if par_couches:
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [K-1, K-1],
                [1 + n[K-1] + j,  1 + n[K-1] + j],
                [1  + n[K-1] + j,  0],
                [1, - (L[K][j] + U[K][j])/2],
            )
        else :
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],
                [0, 0],
                [return_i_from_k_j__variable_z(K,j,n), return_i_from_k_j__variable_z(K,j,n)],
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1, -(L[K][j]+U[K][j])/2],
            )
        # Bornes
        task.putconboundlist(
            [num_contrainte], [mosek.boundkey.up], [-inf], [-L[K][j] * U[K][j]]
        )
        num_contrainte += 1
    return num_contrainte



def contrainte_Mc_Cormick_betai_betaj(task : mosek.Task, 
                                      K : int, 
                                      n : List[int], 
                                      ytrue : int, 
                                      num_contrainte : int, 
                                      par_couches : bool = False):
    # ***** Nombre de contraintes : 3 * (n[K] - 1) * (n[K] - 2) / 2) *****************
    for j in range(n[K]):
        if j == ytrue:
            continue
        jbeta = return_good_j_beta(j, ytrue)
        for i in range(j):
            if i == ytrue:
                continue
            ibeta = return_good_j_beta(i, ytrue)
            # Contrainte : Bij >= betai + betaj - 1 ******************
            if par_couches: 
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte, num_contrainte],
                    [K, K, K],
                    [ibeta, jbeta, jbeta],
                    [0, 0, ibeta],
                    [-0.5, -0.5, 0.5],
                )
            else :
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte, num_contrainte],
                    [1, 1, 1],
                    [ibeta, jbeta, jbeta],
                    [0, 0, ibeta],
                    [-0.5, -0.5, 0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [-1], [inf])
            num_contrainte += 1

            # Contraintes :   Bij <= betai et  Bij <= betaj  ************
            if par_couches  :
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K, K],
                    [jbeta, jbeta],
                    [0, ibeta],
                    [-0.5, 0.5],
                )
            else :
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [1, 1],
                    [jbeta, jbeta],
                    [0, ibeta],
                    [-0.5, 0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [0])
            num_contrainte += 1

            if par_couches:
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K, K],
                    [ibeta, jbeta],
                    [0, ibeta],
                    [-0.5, 0.5],
                )
            else : 
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [1, 1],
                    [ibeta, jbeta],
                    [0, ibeta],
                    [-0.5, 0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [0])
            num_contrainte += 1
    return num_contrainte


def contrainte_Mc_Cormick_betai_betaj_unis(task : mosek.Task, 
                                           K : int, 
                                           n : List[int], 
                                           ytrue : int, 
                                           num_contrainte : int, 
                                           sigmas : bool = False,
                                           par_couches = False,
                                           derniere_couche_lineaire : bool = True):
    # Pas vraiment des contraintes de McCormick ici, on a Bij = 0 pour tous i différent de j
    # ***** Nombre de contraintes : (n[K]-1) * (n[K]-2) / 2 *****************
    if derniere_couche_lineaire:
        sum_n = sum(n)
    else : 
        sum_n = sum(n[:K])
    for j in range(n[K]):
        if j == ytrue:
            continue
        jbeta = return_good_j_beta(j, ytrue)
        for i in range(j):
            if i == ytrue:
                continue
            ibeta = return_good_j_beta(i, ytrue)
            # Contrainte : Bij = 0 ******************
            if par_couches : 
                task.putbarablocktriplet(
                    [num_contrainte],
                    [K-1],
                    [n[K-1] + n[K] +jbeta],
                    [n[K-1] + n[K] +ibeta],
                    [0.5],
                )
            elif not par_couches and not sigmas :
                
                task.putbarablocktriplet(
                    [num_contrainte],
                    [0],
                    [sum_n + jbeta],
                    [sum_n + ibeta],
                    [0.5],
                )
            elif not par_couches and sigmas :
                task.putbarablocktriplet(
                    [num_contrainte],
                    [0],
                    [sum_n + sum(n[1:K]) + jbeta],
                    [sum_n + sum(n[1:K]) + ibeta],
                    [0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
            num_contrainte += 1
    return num_contrainte


def contrainte_Mc_Cormick_betai_zkj(task : mosek.Task, 
                                    K : int, 
                                    n : List[int],
                                    ytrue :int, 
                                    U : List[float], 
                                    num_contrainte : int, 
                                    par_couches = False,
                                    neurones_actifs_stables : List = [],
                                    neurones_inactifs_stables : List = []):
    # Necessite d'avoir les valeurs de la derniere couche dans la matrice, et donc d'avoir la variable derniere_couche_lineaire = True
    # ***** Nombre de contraintes : 2 * (n[K]-1) * sum(n) *****************
    if par_couches : 
        for i in range(n[K]):
            if i == ytrue:
                continue
            ibeta = return_good_j_beta(i, ytrue)
            # Contrainte sur le produit betai * zkj pour k AVANT DERNIERE couche 
            for j in range(n[K-1]):
                if (K-1,j) in neurones_inactifs_stables:
                    task.putbarablocktriplet(
                        [num_contrainte],
                        [K-1],
                        [n[K-1] + n[K] + ibeta],
                        [1 + j],
                        [1/2],
                    )
                    task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
                    num_contrainte += 1
                    continue
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K-1, K-1],
                    [n[K-1] + n[K] + ibeta, n[K-1] + n[K] + ibeta],
                    [1 + j, 0],
                    [1/2, -U[K-1][j]/2],
                )
                task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[K-1][j]], [0])
                num_contrainte += 1

                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K-1, K-1],
                    [n[K-1] + n[K] + ibeta, 1 + j],
                    [1 + j, 0],
                    [1/2, -1/2],
                )
                task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[K-1][j]], [0])
                num_contrainte += 1

            # Contrainte sur le produit betai * zkj pour k DERNIERE couche 
            for j in range(n[K]):
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K-1, K-1],
                    [n[K-1] + n[K] + ibeta, n[K-1] + n[K] + ibeta],
                    [1 + n[K-1] + j, 0],
                    [1/2, -U[K][j]/2],
                )
                task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[K][j]], [0])
                num_contrainte += 1

                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [K-1, K-1],
                    [n[K-1] + n[K] + ibeta, 1 + n[K-1] +  j],
                    [1 + n[K-1] + j, 0],
                    [1.2, -1/2],
                )
                task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[K][j]], [0])
                num_contrainte += 1
    
    else :
        for i in range(n[K]):
            if i == ytrue:
                continue
            ibeta = return_good_j_beta(i, ytrue)
            for k in range(K+1):
                for j in range(n[k]):
                    if (k,j) in neurones_inactifs_stables:
                        task.putbarablocktriplet(
                            [num_contrainte],
                            [0],
                            [sum(n) + ibeta],
                            [return_i_from_k_j__variable_z(k,j,n)],
                            [1/2],
                        )
                        task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
                        num_contrainte += 1
                        continue

                    zj = return_i_from_k_j__variable_z(k,j,n)
                    # Contrainte : betai * zkj <= U betai ******************
                    
                    task.putbarablocktriplet(
                        [num_contrainte, num_contrainte],
                        [0, 0],
                        [sum(n) + ibeta, sum(n) + ibeta],
                        [zj, 0],
                        [0.5, -U[k][j]/2],
                    )
                    task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[k][j]], [0])
                    num_contrainte += 1

                    
                    # Contrainte : 0 <= betai * zkj <= zkj ******************  
                    task.putbarablocktriplet(
                        [num_contrainte,num_contrainte],
                        [0,0],
                        [sum(n) + ibeta, zj],
                        [zj, 0],
                        [0.5, -1/2],
                    )
                    task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [-U[k][j]], [0])
                    num_contrainte += 1
    return num_contrainte



def contrainte_McCormick_zk_sigmak(task : mosek.Task,
                                   K : int,
                                   n : List[int],
                                   U : List[float],
                                   num_contrainte : int):
    """Genere une contrainte de McCormick sur le produit z_(k,j) * sigma_(k,j)"""
    # ***** Nombre de contraintes : 3 * sum(n[1:K]) *****************
    for k in range(1,K):
        for j in range(n[k]):
            
            idx_z = return_i_from_k_j__variable_z(k-1,j,n)
            idx_sigma = return_i_from_k_j__variable_sigma(k,j,n)
            # Contrainte : zkj * sigmakj >= - U sigmakj - zkj - U 
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte,num_contrainte],   
                [0,0,0], 
                [idx_sigma, idx_sigma, idx_z],                   
                [idx_z,0,0],
                [0.5,-U[k][j]/2,-0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.lo],
                            [-U[k][j]],
                            [inf])
            num_contrainte += 1

            # Contrainte :  zkj * sigmakj <= zkj **************
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],   
                [0,0], 
                [idx_sigma, idx_z],                   
                [idx_z,0],
                [0.5,-0.5],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.up],
                            [-inf],
                            [0])
            num_contrainte += 1

            # Contrainte : zkj * sigmakj <= U sigmakj ***********
            task.putbarablocktriplet(
                [num_contrainte, num_contrainte],   
                [0,0], 
                [idx_sigma, idx_sigma],                   
                [idx_z,0],
                [0.5,-U[k][j]/2],
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.up],
                            [-inf],
                            [0])
            num_contrainte += 1
    return num_contrainte



# *******************************************************************    
# ******************* COUPES RLT LAN    *****************************
# *******************************************************************

def coupes_RLT_LAN(task : mosek.Task,
                   K : int,
                   n : List[int],
                   W: List[List[List[float]]],
                   b : List[List[float]],
                   x0 : List[float],
                   epsilon : float,
                   L : List[List[float]],
                   U  : List[List[float]],
                   num_contrainte : int, 
                   par_couches : bool = False,
                   derniere_couche_lineaire : bool = True) :
                    
    # Contrainte 1 : P[xi xi+1] >= P[xi] * Li+1 + P[xi+1] * Li - Li+1 * Li
    # (12b dans l'article LAN2022)
    # ***** Nombre de contraintes : (3 * sum(n[k]*n[k+1] for k in range(K)) + sum(n[1:K])  + 2 * sum((n[k])*(n[k]-1)//2 for k in range(1,K))) *****************
    max_K= K
    if derniere_couche_lineaire:
        max_K -=1
    for k in range(max_K):
        for j1 in range(n[k]):
            for j2 in range(n[k+1]):
                if k==0:
                    lb_j1 = x0[j1] - epsilon
                else:
                    lb_j1 = L[k][j1]
                lb_j2 = L[k+1][j2]

                if par_couches :
                    task.putbarablocktriplet(
                    [num_contrainte, num_contrainte,num_contrainte],   
                    [k,k,k], 
                    [1 + j1, 1 + n[k] + j2, 1 + n[k] + j2],                   
                    [0,0,1 + j1],
                    [-lb_j2/2,-lb_j1/2,0.5],
                    )
                else :
                    idx_j1 = return_i_from_k_j__variable_z(k,j1,n)
                    idx_j2 = return_i_from_k_j__variable_z(k+1,j2,n)
                    task.putbarablocktriplet(
                    [num_contrainte, num_contrainte,num_contrainte],   
                    [0,0,0], 
                    [idx_j1, idx_j2, idx_j2],                   
                    [0,0,idx_j1],
                    [-lb_j2/2,-lb_j1/2,0.5],
                    )
                
                
                
                task.putconboundlist([num_contrainte], [mosek.boundkey.lo],
                                [- lb_j1 * lb_j2],
                                [inf])
                num_contrainte += 1


    # Contrainte 2 : P[xi xi+1] <= P[xi] * Li+1 + P[xi+1] * Ui - Li+1 * Ui
    # (12c - première partie dans l'article LAN2022)
    for k in range(max_K):
        for j1 in range(n[k]):
            for j2 in range(n[k+1]):
                lb_j2 = L[k+1][j2]
                if k==0:
                    ub_j1 = x0[j1] + epsilon
                else:
                    ub_j1 = U[k][j1]
                if par_couches :
                    task.putbarablocktriplet(
                        [num_contrainte, num_contrainte,num_contrainte],   
                        [k,k,k], 
                        [1 + j1, 1 + n[k] + j2, 1 + n[k] + j2],                   
                        [0,0,1 + j1],
                        [-lb_j2/2,-ub_j1/2,0.5],
                        )
                else :
                    idx_j1 = return_i_from_k_j__variable_z(k,j1,n)
                    idx_j2 = return_i_from_k_j__variable_z(k+1,j2,n)
                    task.putbarablocktriplet(
                        [num_contrainte, num_contrainte,num_contrainte],   
                        [0,0,0], 
                        [idx_j1, idx_j2, idx_j2],                   
                        [0,0,idx_j1],
                        [-lb_j2/2,-ub_j1/2,0.5],
                        )
                task.putconboundlist([num_contrainte], [mosek.boundkey.up],
                                [- inf],
                                [- lb_j2 * ub_j1])
                num_contrainte += 1

    # Contrainte 3 : P[xi xi+1] <= P[xi] * Ui+1 + P[xi+1] * Li - Li * Ui+1
    # (12c - deuxième partie article LAN2022)
    for k in range(max_K):
        for j1 in range(n[k]):
            for j2 in range(n[k+1]):
                
                ub_j2 = U[k+1][j2]
                if k==0:
                    lb_j1 = x0[j1] - epsilon
                else:
                    lb_j1 = L[k][j1]
                if par_couches :
                    task.putbarablocktriplet(
                        [num_contrainte, num_contrainte,num_contrainte],   
                        [k,k,k], 
                        [1 + j1, 1 + n[k] + j2, 1 + n[k] + j2],                   
                        [0,0,1 + j1],
                        [-ub_j2/2,-lb_j1/2,0.5],
                        ) 
                else : 
                    idx_j1 = return_i_from_k_j__variable_z(k,j1,n)
                    idx_j2 = return_i_from_k_j__variable_z(k+1,j2,n)
                    task.putbarablocktriplet(
                        [num_contrainte, num_contrainte,num_contrainte],   
                        [0,0,0], 
                        [idx_j1, idx_j2, idx_j2],                   
                        [0,0,idx_j1],
                        [-ub_j2/2,-lb_j1/2,0.5],
                        )
                task.putconboundlist([num_contrainte], [mosek.boundkey.up],
                                [- inf],
                                [- lb_j1 * ub_j2])
                num_contrainte += 1

    # Contrainte 4 : P[xi+1] <= Ai P[xi] + Bi  
    # (Contrainte triangulaire déjà présente dans le modèle de l'article LAN2022)
    for k in range(1, max_K):
        for j in range(n[k]):
            # print(f"Neurone {j} couche {k} : U = {U[k][j]} et L = {L[k][j]}")
            ReLU_L_k_j = L[k][j] 
            ReLU_U_k_j = U[k][j] 
            if L[k][j] < 0:
                ReLU_L_k_j = 0
            if U[k][j] < 0:
                ReLU_U_k_j = 0
            
            k_ = (ReLU_U_k_j - ReLU_L_k_j) / (U[k][j] - L[k][j])
            A = [k_ * W[k-1][j][i] for i in range(n[k-1])]
            B = k_ * (b[k-1][j] - L[k][j]) + ReLU_L_k_j

            if par_couches : 
                A_k = [(1 + i) for i in range(n[k-1])] + [1 + n[k-1] + j]
                A_l = [0] * (n[k - 1] + 1)
                A_v = [-A[i] / 2 for i in range(n[k - 1])] + [1 / 2]

                task.putbarablocktriplet(
                    [num_contrainte] * (len(A_k)), 
                    [k-1] * (len(A_k)),  
                    A_k,  
                    A_l,
                    A_v,)

            else : 
                idx = return_i_from_k_j__variable_z(k - 1, 0, n)
                A_k = [(idx + i) for i in range(n[k - 1])] + [return_i_from_k_j__variable_z(k, j, n)]
                A_l = [0] * (n[k - 1] + 1)
                A_v = [-A[i] / 2 for i in range(n[k - 1])] + [1 / 2]

                task.putbarablocktriplet(
                    [num_contrainte] * (len(A_k)), 
                    [0] * (len(A_k)),  
                    A_k,  
                    A_l,
                    A_v,)
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.up], [-inf], [B]
            )
            num_contrainte += 1


    # Contrainte 5 : P[xi+1 xi+1] <= Ai P[xi xi+1] + Bi P[xi+1]
    # (14 - deuxième partie article LAN2022)
    for k in range(1, max_K):
        for j in range(n[k]):
            for j0 in range(n[k]):
                if j<=j0:
                    continue
                ReLU_L_k_j = L[k][j] 
                ReLU_U_k_j = U[k][j] 
                if L[k][j] < 0:
                    ReLU_L_k_j = 0
                if U[k][j] < 0:
                    ReLU_U_k_j = 0
                
                k_ = (ReLU_U_k_j - ReLU_L_k_j) / (U[k][j] - L[k][j])
                A = [k_ * W[k-1][j][i] for i in range(n[k-1])]
                B = k_ * (b[k-1][j] - L[k][j]) + ReLU_L_k_j

                if par_couches : 
                    A_k = [1 + n[k-1] + j] + [1 + n[k-1] + j0] * (n[k-1] + 1) 
                    A_l = [1 + n[k-1] + j0] + [(1 + i) for i in range(n[k - 1])]   + [0]
                    A_v = [1 / 2] + [-A[i] / 2 for i in range(n[k - 1])]  + [-B/2]

                    task.putbarablocktriplet(
                        [num_contrainte] * (len(A_k)), 
                        [k-1] * (len(A_k)),  
                        A_k,  
                        A_l,
                        A_v,)

                else : 
                    idx = return_i_from_k_j__variable_z(k - 1, 0, n)
                    A_k = [return_i_from_k_j__variable_z(k, j, n)] + [return_i_from_k_j__variable_z(k,j0,n)] * (n[k-1] + 1) 
                    A_l = [return_i_from_k_j__variable_z(k,j0,n)] + [(idx + i) for i in range(n[k - 1])]   + [0]
                    A_v = [1 / 2] + [-A[i] / 2 for i in range(n[k - 1])]  + [-B/2]
                    task.putbarablocktriplet(
                        [num_contrainte] * (len(A_k)), 
                        [0] * (len(A_k)),  
                        A_k,  
                        A_l,
                        A_v,)
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.up], [-inf], [0]
                )
                num_contrainte += 1

    # Contrainte 6 : P[xi+1 xi+1] <= P[xi+1] Ui+1
    # (14 - première partie article LAN2022)
    for k in range(1, max_K):
        for j in range(n[k]):
            for j0 in range(n[k]):
                if j<=j0:
                    continue
                if par_couches : 
                    A_k = [1 + n[k-1] + j] + [1 + n[k-1] + j]  
                    A_l = [1 + n[k-1] + j0] + [0]
                    A_v = [1 / 2] + [-U[k][j0]/2]

                    task.putbarablocktriplet(
                        [num_contrainte] * (len(A_k)), 
                        [k-1] * (len(A_k)),  
                        A_k,  
                        A_l,
                        A_v,)

                else : 
                    idx = return_i_from_k_j__variable_z(k - 1, 0, n)
                    A_k = [return_i_from_k_j__variable_z(k, j, n)] + [return_i_from_k_j__variable_z(k,j,n)] 
                    A_l = [return_i_from_k_j__variable_z(k,j0,n)] + [0]  
                    A_v = [1 / 2] + [-U[k][j0]/2]

                    task.putbarablocktriplet(
                        [num_contrainte] * (len(A_k)), 
                        [0] * (len(A_k)),  
                        A_k,  
                        A_l,
                        A_v,)
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.up], [-inf], [0]
                )
                num_contrainte += 1

    return num_contrainte