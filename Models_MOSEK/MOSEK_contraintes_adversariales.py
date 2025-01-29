import numpy as np
import mosek
from typing import List, Tuple
from MOSEK_outils import return_good_j_beta, return_i_from_k_j__variable_z, return_k_j_from_i
inf = 10e5





# ************ Contraintes adversariales ***************************
# Contrainte definissant un exemple adverse par les betas  

def contrainte_exemple_adverse_beta_u(
        task : mosek.Task,
        K : int,
        n : List[int],
        W : List[List[List[float]]],
        ytrue : int,
        U : List[List[float]],
        rho : float,
        num_contrainte : int, 
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True):
    # Normalement : cas betas_z_unis = False
    # ***** Contrainte : U (1 - betaj) + zKj > zKj*  *****
    # ***** Nombre de contraintes : n[K] - 1 *****************
    for j in range(n[K]):
        if j == ytrue:
            continue
        jbeta = return_good_j_beta(j, ytrue)
        if par_couches and derniere_couche_lineaire: 
            # Partie Beta
            A5_beta_k = [jbeta]
            A5_beta_l = [0]
            A5_beta_v = [-U[K][j] / 2]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A5_beta_k)),  
                [K] * (len(A5_beta_k)), 
                A5_beta_k,  
                A5_beta_l,
                A5_beta_v,
            )
            # Partie z
            A5_z_k = [1+n[K-1]+j, 1+n[K-1]+ytrue]
            A5_z_l = [0, 0]
            A5_z_v = [1 / 2, -(1 + rho) / 2]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A5_z_k)), 
                [K-1] * (len(A5_z_k)),  
                A5_z_k,  
                A5_z_l,
                A5_z_v,
            )
        elif not par_couches and derniere_couche_lineaire : 
            # Partie Beta
            task.putbarablocktriplet(
                [num_contrainte], 
                [1],  
                [jbeta], 
                [0],
                [-U[K][j] / 2],
            )
            # Partie z
            A5_z_k = [return_i_from_k_j__variable_z(K, j, n), return_i_from_k_j__variable_z(K, ytrue, n)]
            A5_z_l = [0, 0]
            A5_z_v = [1 / 2, -(1 + rho) / 2]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A5_z_k)),  
                [0] * (len(A5_z_k)),  
                A5_z_k,  
                A5_z_l,
                A5_z_v,
            )

        elif par_couches and not derniere_couche_lineaire:
            poids_W = []
            for i in range(n[K-1]):
                poids_W.append( (W[K-1][j][i] - (1 + rho)*W[K-1][ytrue][i]) /2)
            # Partie Beta
            task.putbarablocktriplet(
                [num_contrainte],  
                [K-1], 
                [jbeta],  
                [0],
                [-U[K][j] / 2],
            )
            # Partie z
            task.putbarablocktriplet(
                [num_contrainte] * n[K-1], 
                [K-2] * n[K-1] ,  
                [(1+n[K-2]+i) for i in range(n[K-1])],  
                [0] * n[K-1],
                poids_W,
            ) 

        elif not par_couches and not derniere_couche_lineaire:
            poids_W = []
            for i in range(n[K-1]):
                poids_W.append( (W[K-1][j][i] - (1 + rho)*W[K-1][ytrue][i]) /2)
            
            # Partie Beta
            task.putbarablocktriplet(
                [num_contrainte],  
                [1], 
                [jbeta],  
                [0],
                [-U[K][j] / 2],
            )
            # Partie z
            task.putbarablocktriplet(
                [num_contrainte] * n[K-1], 
                [0] * n[K-1],  
                [(1+sum(n[:(K-1)])+i) for i in range(n[K-1])],  
                [0] * n[K-1],
                poids_W,
            )       


        # Bornes
        task.putconboundlist(
            [num_contrainte], [mosek.boundkey.lo], [-U[K][j]], [inf]
        )
        num_contrainte += 1
    return num_contrainte



def contrainte_exemple_adverse_beta_produit_simple(
        task : mosek.Task,
        K : int,
        n : List[int],
        W : List[List[List[float]]],
        ytrue : int,
        U : List[List[float]],
        rho : float,
        num_contrainte : int, 
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True):
    # ***** Contrainte : betaj zKj > betaj zKj*  *****
    # On a obligatoirement betas_z_unis = True
    # ***** Nombre de contraintes : n[K] - 1 *****************
    for j in range(n[K]):
        if j == ytrue:   
            continue
        jbeta = return_good_j_beta(j, ytrue)

        if par_couches and derniere_couche_lineaire: 
            A5_k = [n[K-1] + n[K] + jbeta, n[K-1] + n[K] + jbeta]
            A5_l = [1 + n[K-1] + j, 1 + n[K-1] + ytrue]
            A5_v = [1/2, -1/ 2]
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [K-1] * 2, 
                A5_k,  
                A5_l,
                A5_v,
            )
        elif not par_couches and derniere_couche_lineaire : 
            idx_ytrue = return_i_from_k_j__variable_z(K,ytrue,n)
            idx_j = return_i_from_k_j__variable_z(K,j,n)
            task.putbarablocktriplet(
                [num_contrainte] * 2, 
                [0] * 2,  
                [sum(n) + jbeta] * n[K-1] * 2, 
                [(1+sum(n[:K])+i) for i in range(n[K-1])],
                [(W[K-1][j][i]/2) for i in range(n[K-1])] + [(W[K-1][ytrue][i]/2) for i in range(n[K-1])] ,
            )

        elif par_couches and not derniere_couche_lineaire: 
            poids_W = []
            for i in range(n[K-1]):
                poids_W.append(W[K-1][j][i]-(1+rho)*W[K-1][ytrue][i])
            task.putbarablocktriplet(
                [num_contrainte] * n[K-1],  
                [K-2] * n[K-1], 
                [n[K-2]+n[K-1]+jbeta] * n[K-1],  
                [(1+n[K-2]+i) for i in range(n[K-1])],
                poids_W,
            )

        # Bornes
        task.putconboundlist(
            [num_contrainte], [mosek.boundkey.lo], [0], [inf]
        )
        num_contrainte += 1
    return num_contrainte


def contrainte_exemple_adverse_beta_produit_complet(
        task : mosek.Task,
        K : int,
        n : List[int],
        W : List[List[List[float]]],
        ytrue : int,
        num_contrainte : int, 
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True):
    # ***** Contrainte : betaj zKj > betaj zKj2 pour tout j2  *****
    # On a obligatoirement betas_z_unis = True
    # ***** Nombre de contraintes : (n[K] - 1) * (n[K] - 1) *****************
    for j in range(n[K]):
        if j == ytrue:   
            continue
        jbeta = return_good_j_beta(j, ytrue)
        for j2 in range(n[K]):
            if j2 == ytrue or j2==j:   
                continue
            if par_couches and derniere_couche_lineaire: 
                A5_k = [1 + n[K-1] + n[K] + jbeta, 1 + n[K-1] + n[K] + jbeta]
                A5_l = [1 + n[K-1] + j, 1 + n[K-1] + j2]
                A5_v = [1/2, -1/ 2]
                task.putbarablocktriplet(
                    [num_contrainte] * 2,  
                    [K-1] * 2, 
                    A5_k,  
                    A5_l,
                    A5_v,
                )
            elif not par_couches and derniere_couche_lineaire: 
                idx_j2 = return_i_from_k_j__variable_z(K,j2,n)
                idx_j = return_i_from_k_j__variable_z(K,j,n)
                A5_k = [sum(n) + jbeta, sum(n) + jbeta]
                A5_l = [idx_j, idx_j2]
                A5_v = [1/2, -1/ 2]
                task.putbarablocktriplet(
                    [num_contrainte] * 2, 
                    [0] * 2,  
                    A5_k, 
                    A5_l,
                    A5_v,
                )

            elif par_couches and not derniere_couche_lineaire: 
                poids_Wyj2 = [(-W[j2][i] / 2) for i in range(n[K-1])]
                poids_Wyj = [(W[j][i] / 2) for i in range(n[K-1])]
                task.putbarablocktriplet(
                    [num_contrainte] * 2 * n[K-1],  
                    [K-2] * 2 * n[K-1], 
                    [1 + n[K-2]  + n[K-1] + j] * 2 * n[K-1],  
                    [(1 + n[K-2] + i) for i in range(n[K-1])] * 2,
                    poids_Wyj + poids_Wyj2 ,
                )
            elif not par_couches and not derniere_couche_lineaire: 
                poids_Wyj2 = [(-W[j2][i] / 2) for i in range(n[K-1])]
                poids_Wyj = [(W[j][i] / 2) for i in range(n[K-1])]
                task.putbarablocktriplet(
                    [num_contrainte] * 2 * n[K-1], 
                    [0] * 2 * n[K-1],  
                    [(1+sum(n[:K])+j)] * 2 * n[K-1], 
                    [(1+sum(n[:(K-1)])+i) for i in range(n[K-1])] * 2,
                    poids_Wyj + poids_Wyj2,
                )


            # Bornes
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.lo], [0], [inf]
            )
            num_contrainte += 1
    return num_contrainte


def contrainte_exemple_adverse_somme_beta_egale_1(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        num_contrainte : int, 
        par_couches : bool = False, 
        betas_z_unis : bool = False,
        derniere_couche_lineaire : bool = True):
    # *****  Contrainte : somme(betaj) == 1 ******
    # ***** Nombre de contraintes :1 *****************
    if par_couches and betas_z_unis and derniere_couche_lineaire :
        A6_k = [(n[K-1] + n[K] + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue]
        A6_l = [0 for i in range(n[K] - 1)]
        A6_v = [1 / 2 for i in range(n[K] - 1)]
        task.putbarablocktriplet(
            [num_contrainte] * (len(A6_k)),  
            [K-1] * (len(A6_k)),  
            A6_k, 
            A6_l,
            A6_v,
        )
    elif par_couches and not betas_z_unis and derniere_couche_lineaire:
        A6_k = [(return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue]
        A6_l = [0 for i in range(n[K] - 1)]
        A6_v = [1 / 2 for i in range(n[K] - 1)]
        task.putbarablocktriplet(
            [num_contrainte] * (len(A6_k)),  
            [K] * (len(A6_k)),  
            A6_k, 
            A6_l,
            A6_v,
        )
    elif not par_couches and betas_z_unis and derniere_couche_lineaire:
        A6_k = [(sum(n) + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue]
        A6_l = [0 for i in range(n[K] - 1)]
        A6_v = [1 / 2 for i in range(n[K] - 1)]
        task.putbarablocktriplet(
            [num_contrainte] * (len(A6_k)),  
            [0] * (len(A6_k)),  
            A6_k, 
            A6_l,
            A6_v,
        )
    elif not par_couches and not betas_z_unis :
        task.putbarablocktriplet(
            [num_contrainte] * (n[K] - 1),  
            [1] * (n[K] - 1),  
            [(return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K]-1),
            [1/2] * (n[K]-1),
        )
    # *******************************
    elif par_couches and betas_z_unis and not derniere_couche_lineaire :
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]-1),  
            [K-2] * (n[K]-1),  
            [(sum(n[:K])+return_good_j_beta(j,ytrue)) for j in range(n[K]) if j!=ytrue], 
            [0] * (n[K]-1),
            [1 / 2 for i in range(n[K] - 1)],
        )
    elif par_couches and not betas_z_unis and not derniere_couche_lineaire:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]-1),  
            [K] * (n[K]-1),  
            [(return_good_j_beta(j,ytrue)) for j in range(n[K-1])], 
            [0] * (n[K]-1),
            [1 / 2 for i in range(n[K] - 1)],
        )


    elif not par_couches and betas_z_unis and not derniere_couche_lineaire:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]-1),  
            [0] * (n[K] - 1),  
            [(1+sum(n[:K]+return_good_j_beta(i,ytrue))) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K] - 1),
            [1 / 2 for i in range(n[K] - 1)],
        )

    # Bornes
    task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [1], [1])
    num_contrainte += 1
    return num_contrainte


def contrainte_exemple_adverse_somme_beta_superieure_1(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        num_contrainte : int, 
        par_couches : bool = False,
        betas_z_unis : bool = False,
        derniere_couche_lineaire : bool = False):
    # *****  Contrainte : somme(betaj) > 1 ******
    # ***** Nombre de contraintes : 1 *****************
    if par_couches and betas_z_unis and derniere_couche_lineaire:
        A6_k = [(n[K-1] + n[K] + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue]
        A6_l = [0 for i in range(n[K] - 1)]
        A6_v = [1 / 2 for i in range(n[K] - 1)]
        task.putbarablocktriplet(
            [num_contrainte] * (len(A6_k)),  
            [K-1] * (len(A6_k)),  
            A6_k, 
            A6_l,
            A6_v,
        )
    elif par_couches and betas_z_unis and not derniere_couche_lineaire:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]- 1),  
            [K-2] * (n[K]-1),  
            [(n[K-2] + n[K-1] + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K] - 1),
            [1 / 2] * (n[K] - 1),
    )
    elif par_couches and not betas_z_unis:
        A6_k = [(return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue]
        A6_l = [0] * (n[K] - 1)
        A6_v = [1 / 2] * (n[K] - 1)
        task.putbarablocktriplet(
            [num_contrainte] * (len(A6_k)),  
            [K-1] * (len(A6_k)),  
            A6_k, 
            A6_l,
            A6_v,
        )

    elif not par_couches and betas_z_unis and derniere_couche_lineaire:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]-1),  
            [0] * (n[K]-1),  
            [(sum(n) + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K] - 1),
            [1/2] * (n[K] - 1),
        )
    elif not par_couches and betas_z_unis and not derniere_couche_lineaire:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K]-1),  
            [0] * (n[K]-1),  
            [(sum(n[:K]) + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K] - 1),
            [1/2] * (n[K] - 1),
        )
    elif not par_couches and not betas_z_unis:
        task.putbarablocktriplet(
            [num_contrainte] * (n[K] - 1),  
            [1] * (n[K] - 1),  
            [(return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue], 
            [0] * (n[K] - 1),
            [1/2] * (n[K] - 1),
        )
    # Bornes
    task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [1], [inf])
    num_contrainte += 1
    return num_contrainte


def contrainte_beta_discret(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        num_contrainte : int, 
        par_couches : bool = False, 
        betas_z_unis : bool = False,
        derniere_couche_lineaire : bool = False):
    # ******** Contrainte : betaj = betaj^2 ***************
    # ***** Nombre de contraintes : n[K] - 1 *****************
    for j in range(n[K]):
        if j == ytrue:
            continue
        jbeta = return_good_j_beta(j, ytrue)
        if par_couches and betas_z_unis and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [K-1] * 2,  
                [n[K] + n[K-1] + jbeta, n[K] + n[K-1] + jbeta],  
                [0, n[K] + n[K-1] + jbeta],
                [1 / 2, -1],
            )
        elif par_couches and betas_z_unis and not derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [K-2] * 2,  
                [n[K-2] + n[K-1] + jbeta, n[K-2] + n[K-1] + jbeta],  
                [0, n[K-2] + n[K-1] + jbeta],
                [1 / 2, -1],
            )
        elif par_couches and not betas_z_unis : 
            numvar = K-1
            if derniere_couche_lineaire :
                numvar -= 1
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [numvar] * 2,  
                [jbeta, jbeta],  
                [0, jbeta],
                [1 / 2, -1],
            )
        
        elif not par_couches and betas_z_unis :
            sum_n = sum(n)
            if derniere_couche_lineaire:
                sum_n -= n[K]
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum_n + jbeta, sum_n + jbeta],  
                [0, sum_n + jbeta],
                [1 / 2, -1],
            )
        elif not par_couches and not betas_z_unis: 
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [1] * 2,  
                [jbeta, jbeta],  
                [0, jbeta],
                [1 / 2, -1],
            )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
        num_contrainte += 1
    return num_contrainte




def contrainte_borne_betas(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        num_contrainte : int, 
        sigmas : bool = False,
        par_couches : bool = False, 
        betas_z_unis : bool = False,
        derniere_couche_lineaire : bool = True):
    # *****  Contrainte : 0 <= betaj <= 1 ******
    # ***** Nombre de contraintes : n[K] - 1 *****************
    for i in range(n[K]):
        if i==ytrue:
            continue
        if par_couches and betas_z_unis and derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte],  
                [K-1],  
                [n[K-1] + n[K] + return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )
        elif par_couches and betas_z_unis and not derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte],  
                [K-2],  
                [n[K-2] + n[K-1] + return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )
        elif par_couches and not betas_z_unis :
            numvar = K
            if derniere_couche_lineaire :
                numvar -= 1
            task.putbarablocktriplet(
                [num_contrainte],  
                [numvar],  
                [return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )

        elif not par_couches and betas_z_unis and not sigmas :
            sum_n = sum(n)
            if derniere_couche_lineaire:
                sum_n -= 1
            task.putbarablocktriplet(
                [num_contrainte],  
                [0],  
                [sum_n + return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )
        elif not par_couches and betas_z_unis and sigmas :
            sum_n = sum(n)
            if derniere_couche_lineaire :
                sum_n -= 1
            task.putbarablocktriplet(
                [num_contrainte],  
                [0],  
                [sum_n + sum(n[1:K]) + return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )
        elif not par_couches and not betas_z_unis :
            task.putbarablocktriplet(
                [num_contrainte],  
                [1],  
                [return_good_j_beta(i, ytrue)], 
                [0],
                [1/2],
            )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [0], [1])
        num_contrainte += 1
    return num_contrainte







def contrainte_borne_betas_unis(
        task : mosek.Task,
        K : int,
        n : List[int],
        W : List[List[List[float]]],
        ytrue : int,
        U : List[List[float]],
        L : List[List[float]],
        rho : float,
        num_contrainte : int, 
        sigmas : bool = False,
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True):
    # On a forcément : betas_z_unis = True
    # *** Ce sont des contraintes de McCormick ******************
    # ***** Nombre de contraintes : 4 * (n[K] - 1) *****************

    # *****  Contrainte : betaj zK_j - zK_j - betaj U >= -U **************
    for j in range(n[K]):
        if j==ytrue:
            continue
        if par_couches and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [K-1] * 3,  
                [n[K-1] + n[K] + return_good_j_beta(j, ytrue), 1 + n[K-1] + j, n[K-1] + n[K] + return_good_j_beta(j, ytrue)], 
                [1 + n[K-1] + j, 0, 0],
                [1/2, -1/2, -U[K][j]/2],
            )
        elif par_couches and not derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * (2*n[K-1]+1),  
                [K-2] * (2*n[K-1]+1),  
                [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)]*n[K-1] + [(1+n[K-2]+i) for i in range(n[K-1])] + [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)], 
                [(1+n[K-2]+i) for i in range(n[K-1])] + [0]*n[K-1] + [0],
                [(W[K-1][j][i]/2) for i in range (n[K-1])]*2 + [-U[K][j]/2],
            )
        
        elif not par_couches and not sigmas and derniere_couche_lineaire : 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [0] * 3,  
                [sum(n) + return_good_j_beta(j, ytrue), return_i_from_k_j__variable_z(K,j,n), sum(n) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0, 0],
                [1/2, -1/2, -U[K][j]/2],
            )
        elif not par_couches and not sigmas and not derniere_couche_lineaire : 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [0] * 3,  
                [sum(n[:K]) + return_good_j_beta(j, ytrue)] * n[K-1] + [(1+sum(n[:K]+i)) for i in range(n[K-1])] + [sum(n) + return_good_j_beta(j, ytrue)], 
                [(1+sum(n[:K]+i)) for i in range(n[K-1])] + [0] * n[K-1] + [0],
                [(W[K-1][j][i]/2) for i in range (n[K-1])]*2 + [U[K][j]/2],
            )

        elif not par_couches and sigmas and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [0] * 3,  
                [sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue), 
                 return_i_from_k_j__variable_z(K,j,n), 
                 sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0, 0],
                [1/2, -1/2, -U[K][j]/2],
            )
    
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [-U[K][j]], [inf])
        num_contrainte += 1

    # *****  Contrainte : betaj zK_j <= zK_j - L + betaj L **************
    for j in range(n[K]):
        if j==ytrue:
            continue
        if par_couches and derniere_couche_lineaire : 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [K-1] * 3,  
                [n[K-1] + n[K] + return_good_j_beta(j, ytrue), 
                 1 + n[K-1] + j, 
                 n[K-1] + n[K] + return_good_j_beta(j, ytrue)], 
                [1 + n[K-1] + j, 0, 0],
                [1/2, -1/2, -L[K][j]/2],
            )
        elif par_couches and not derniere_couche_lineaire : 
            task.putbarablocktriplet(
                [num_contrainte] * (2*n[K-1]+1),  
                [K-2] * (2*n[K-1]+1),  
                [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)]*n[K-1] + 
                    [(1+ n[K-2]+ i) for i in range(n[K-1])] + 
                    [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)], 
                [(1+ n[K-2]+ i) for i in range(n[K-1])] +  [0]*n[K-1] + [0],
                [(W[K-1][j][i]/2) for i in range(n[K-1])]*2 + [-L[K][j]/2],
            )
        elif not par_couches and not sigmas and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [0] * 3,  
                [sum(n) + return_good_j_beta(j, ytrue), return_i_from_k_j__variable_z(K,j,n), sum(n) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0, 0],
                [1/2, -1/2, -L[K][j]/2],
            )
        elif not par_couches and not sigmas and not derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * (2*n[K-1]+1),  
                [0] * (2*n[K-1]+1),  
                [sum(n[:K]) + return_good_j_beta(j, ytrue)]*n[K-1] + [(1+sum(n[:K])+i) for i in range(n[K-1])] + [sum(n) + return_good_j_beta(j, ytrue)], 
                [(1+sum(n[:K])+i) for i in range(n[K-1])] + [0]*n[K-1] + [0],
                [(W[K-1][j][i]/2) for i in range(n[K-1])]*2 + [-L[K][j]/2],
            )

        elif not par_couches and sigmas and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 3,  
                [0] * 3,  
                [sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue), 
                 return_i_from_k_j__variable_z(K,j,n), 
                 sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0, 0],
                [1/2, -L[K][j]/2, -1/2],
            )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [-L[K][j]])
        num_contrainte += 1

    
    # *****  Contrainte : zK_j betaj <= U betaj **************
    for j in range(n[K]):
        if j==ytrue:
            continue
        if par_couches and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [K-1] * 2,  
                [n[K-1] + n[K] + return_good_j_beta(j, ytrue), n[K-1] + n[K] + return_good_j_beta(j, ytrue)], 
                [1 + n[K-1] + j, 0],
                [1/2, -U[K][j]/2],
            )
        elif par_couches and not derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * (n[K-1] + 1),  
                [K-2] * (n[K-1]+1),  
                [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)]*n[K-1] +  [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)], 
                [(1 +n[K-2] +i) for i in range(n[K-1])] + [0],
                [(W[K-1][j][i]/2) for i in range(n[K-1])] + [-U[K][j]/2],
            )
        elif not par_couches and not sigmas and derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum(n) + return_good_j_beta(j, ytrue), sum(n) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1/2, -U[K][j]/2],
            )
        elif not par_couches and not sigmas and not derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte] * (n[K-1]+1),  
                [0] * (n[K-1]+1),  
                [sum(n[:K]) + return_good_j_beta(j, ytrue)]*n[K-1] + [sum(n[:K]) + return_good_j_beta(j, ytrue)], 
                [(1+sum(n[:K])+i) for i in range(n[K-1])] + [0],
                [(W[K-1][j][i]/2) for i in range(n[K-1])] +  [-U[K][j]/2],
            )
        elif not par_couches and sigmas and derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue), sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1/2, -U[K][j]/2],
            )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [0])
        num_contrainte += 1

    # *****  Contrainte : zK_j betaj >= L betaj **************
    for j in range(n[K]):
        if j==ytrue:
            continue
        if par_couches and derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [K-1] * 2,  
                [n[K-1] + n[K] + return_good_j_beta(j, ytrue), n[K-1] + n[K] + return_good_j_beta(j, ytrue)], 
                [1 + n[K-1] + j, 0],
                [1/2, -L[K][j]/2],
            )
        if par_couches and not derniere_couche_lineaire: 
            task.putbarablocktriplet(
                [num_contrainte] * (n[K-1] + 1),  
                [K-2] * (n[K-1] + 1),  
                [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)]*n[K-1] + [n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)], 
                [(1 +n[K-2] +i) for i in range(n[K-1])] +  [0],
                [(W[K-1][j][i]/2) for i in range(n[K-1])] + [-L[K][j]/2],
            )
        elif not par_couches and not sigmas and derniere_couche_lineaire :
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum(n) + return_good_j_beta(j, ytrue), sum(n) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1/2, -L[K][j]/2],
            )
        elif not par_couches and not sigmas and not derniere_couche_lineaire :
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum(n[:K]) + return_good_j_beta(j, ytrue)] + [sum(n[:K]) + return_good_j_beta(j, ytrue)], 
                [(1+sum(n[:K])+i) for i in range(n[K-1])] + [0],
                [1/2] + [-L[K][j]/2],
            )
        elif not par_couches and sigmas and derniere_couche_lineaire:
            task.putbarablocktriplet(
                [num_contrainte] * 2,  
                [0] * 2,  
                [sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue), sum(n) + sum(n[1:K]) + return_good_j_beta(j, ytrue)], 
                [return_i_from_k_j__variable_z(K,j,n), 0],
                [1/2, -L[K][j]/2],
            )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [0], [inf])
        num_contrainte += 1
    return num_contrainte



def contrainte_produit_betas_nuls_Adv2_Adv3(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        num_contrainte : int, 
        sigmas : bool = False,
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True,
):
    # **** Contrainte : betai*betaj = 0
    # ***** Nombre de contraintes : (n[K] - 1) * (n[K] - 2) / 2 *****************
    if par_couches :
        return num_contrainte
    else : 
        for i in range(n[K]) :
            if i== ytrue :
                continue
            for j in range(i):
                if j == ytrue : 
                    continue
                if derniere_couche_lineaire :
                    task.putbarablocktriplet(
                            [num_contrainte],  
                            [0] ,  
                            [sum(n) + return_good_j_beta(i, ytrue)], 
                            [sum(n) + return_good_j_beta(j, ytrue)],
                            [1/2],
                        )
                else : 
                    task.putbarablocktriplet(
                            [num_contrainte],  
                            [0] ,  
                            [sum(n[:K]) + return_good_j_beta(i, ytrue)], 
                            [sum(n[:K]) + return_good_j_beta(j, ytrue)],
                            [1/2],
                        )
                task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
                num_contrainte += 1
    return num_contrainte


def contrainte_borne_somme_betaz(
        task : mosek.Task,
        K : int,
        n : List[int],
        W : List[List[List[float]]],
        ytrue : int,
        U : List[List[float]],
        num_contrainte : int, 
        sigmas : bool = False,
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True,
):
    poids_W = []
    for j in range(n[K]):
        if j!=ytrue:
            poids_W.extend([(W[K-1][j][i]/2) for i in range(n[K-1])] )
    if par_couches and derniere_couche_lineaire : 
        task.putbarablocktriplet(
                [num_contrainte] * (n[K] - 1),  
                [K-1] * (n[K] - 1),  
                [(n[K-1] + n[K] + return_good_j_beta(j, ytrue)) for j in range(n[K]) if j!= ytrue], 
                [(1 + n[K-1] + j) for j in range(n[K]) if j!= ytrue],
                [1/2] * (n[K] - 1),
            )
    elif par_couches and not derniere_couche_lineaire : 
        task.putbarablocktriplet(
                [num_contrainte] * (n[K] - 1) * n[K-1],  
                [K-2] * (n[K] - 1) * n[K-1],  
                [(n[K-2] + n[K-1] + return_good_j_beta(j, ytrue)) for j in range(n[K]) if j!= ytrue for i in range(n[K-1])], 
                [(1 + n[K-2] + i) for i in range(n[K-1])] * (n[K] -  1),
                poids_W,
            )
    elif not par_couches and sigmas and derniere_couche_lineaire:
        task.putbarablocktriplet(
                [num_contrainte] * (n[K] - 1),  
                [0] * (n[K] - 1),  
                [(sum(n) + sum(n[1:K])+ return_good_j_beta(j, ytrue)) for j in range(n[K]) if j!=ytrue], 
                [return_i_from_k_j__variable_z(K,j,n) for j in range(n[K]) if j!=ytrue],
                [1/2] * (n[K] - 1),
            ) 
    elif not par_couches and not sigmas and derniere_couche_lineaire: 
        task.putbarablocktriplet(
                [num_contrainte] * (n[K] - 1),  
                [0] * (n[K] - 1),  
                [(sum(n) + return_good_j_beta(j, ytrue)) for j in range(n[K]) if j!=ytrue], 
                [return_i_from_k_j__variable_z(K,j,n) for j in range(n[K]) if j!=ytrue],
                [1/2] * (n[K] - 1),
            )
    elif not par_couches and not sigmas and not derniere_couche_lineaire: 
        task.putbarablocktriplet(
                [num_contrainte] * (n[K] - 1) * n[K-1],  
                [0] * (n[K] - 1) * n[K-1],  
                [(sum(n[:K]) + return_good_j_beta(j, ytrue)) for i in range(n[K-1]) for j in range(n[K]) if j!=ytrue], 
                [(1+sum(n[:K])+i) for i in range(n[K-1])] * (n[K] - 1),
                poids_W,
            )
    # Bornes
    task.putconboundlist([num_contrainte], [mosek.boundkey.up], [-inf], [max(U[K])])
    num_contrainte += 1
    return num_contrainte

