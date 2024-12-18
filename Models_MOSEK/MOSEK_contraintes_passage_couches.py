import numpy as np
import mosek
from typing import List, Tuple
from MOSEK_outils import (
    return_i_from_k_j__variable_z,
    return_i_from_k_j__variable_sigma,
    return_k_j_from_i,
)

inf = 10e5


# ********** Contrainte de borne sur les couches internes ***********
def contrainte_borne_couches_internes(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        U : List[List[float]], 
        num_contrainte : int, 
        par_couches : bool = False):
    # ***** Nombre de contraintes : sum(n[1:K]) *****************
    if par_couches:
        for k in range(1, K):
            for j in range(n[k]):
                task.putbarablocktriplet(
                    [num_contrainte],
                    [k-1],
                    [1 + n[k - 1] + j],
                    [0],
                    [1 / 2],
                )
                # Bornes
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.ra], [0], [U[k][j]]
                )
                num_contrainte += 1
    else:
        for k in range(1, K):
            for j in range(n[k]):
                task.putbarablocktriplet(
                    [num_contrainte],
                    [0],
                    [return_i_from_k_j__variable_z(k, j, n)],
                    [0],
                    [1 / 2],
                )
                # Bornes
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.ra], [0], [U[k][j]]
                )
                num_contrainte += 1
    return num_contrainte


# ********** Contrainte de borne quadratique sur les couches internes et la couche initiale ***********
def contrainte_quadratique_bornes(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        L : List[List[float]], 
        U : List[List[float]], 
        x0 : List[float], 
        epsilon : float, 
        num_contrainte : int, 
        par_couches : bool =False):
    # ***** Nombre de contraintes : sum(n[0:K]) *****************
    if par_couches:
        lb = np.maximum(x0 - epsilon * np.ones(len(x0)), L[0])
        ub = np.minimum(x0 + epsilon * np.ones(len(x0)), U[0])
        for j in range(n[0]):
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [0, 0],
                    [1 + j, 1 + j],
                    [1 + j, 0],
                    [1, - (lb[j]+ub[j]) / 2],
                )
                # Bornes
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.up], [-inf], [-lb[j] * ub[j]]
                )
                num_contrainte += 1

        for k in range(1,K):
            for j in range(n[k]):
                lb = L[k]
                ub = U[k]
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [k-1, k-1],
                    [1 + n[k - 1] + j, 1 + n[k - 1] + j],
                    [1 + n[k - 1] + j, 0],
                    [1, - (lb[j]+ub[j]) / 2],
                )
                # Bornes
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.up], [-inf], [-lb[j] * ub[j]]
                )
                num_contrainte += 1
    else:
        for k in range(K):
            if k == 0:
                lb = np.maximum(x0 - epsilon * np.ones(len(x0)), L[0])
                ub = np.minimum(x0 + epsilon * np.ones(len(x0)), U[0])
            else : 
                lb = L[k]
                ub = U[k]
            for j in range(n[k]):
                task.putbarablocktriplet(
                    [num_contrainte, num_contrainte],
                    [0, 0],
                    [return_i_from_k_j__variable_z(k, j, n), return_i_from_k_j__variable_z(k, j, n)],
                    [return_i_from_k_j__variable_z(k, j, n), 0],
                    [1, -(lb[j]+ub[j])/2],
                )
                # Bornes
                task.putconboundlist(
                    [num_contrainte], [mosek.boundkey.up], [-inf], [-lb[j] * ub[j]]
                )
                num_contrainte += 1
    return num_contrainte


# ****** Contrainte lineaire sur la derniere couche **************
def contrainte_derniere_couche_lineaire(
    task : mosek.Task, 
    K : int, 
    n : List[int], 
    W : List[List[List[float]]], 
    b : List[List[float]], 
    num_contrainte : int, 
    par_couches : bool = False
):
    # Contrainte : zK = WK-1 * zK-1 + bK-1
    # ***** Nombre de contraintes : n[K] *****************
    if par_couches:
        for j in range(n[K]):
            A4_k = [(1 + i) for i in range(n[K - 1])] + [1 + n[K - 1] + j]
            A4_l = [0] * (n[K - 1] + 1)
            A4_v = [-W[K - 1][j][i] / 2 for i in range(n[K - 1])] + [1 / 2]

            task.putbarablocktriplet(
                [num_contrainte] * (len(A4_k)),
                [K - 1] * (len(A4_k)),
                A4_k,
                A4_l,
                A4_v,
            )
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.fx], [b[K - 1][j]], [b[K - 1][j]]
            )
            num_contrainte += 1
    else:
        for j in range(n[K]):
            idx = return_i_from_k_j__variable_z(K - 1, 0, n)
            A4_k = [(idx + i) for i in range(n[K - 1])] + [
                return_i_from_k_j__variable_z(K, j, n)
            ]
            A4_l = [0] * (n[K - 1] + 1)
            A4_v = [-W[K - 1][j][i] / 2 for i in range(n[K - 1])] + [1 / 2]

            task.putbarablocktriplet(
                [num_contrainte] * (len(A4_k)),
                [0] * (len(A4_k)),
                A4_k,
                A4_l,
                A4_v,
            )
            task.putconboundlist(
                [num_contrainte], [mosek.boundkey.fx], [b[K - 1][j]], [b[K - 1][j]]
            )
            num_contrainte += 1

    return num_contrainte


# *********************************************************
# ************* CONTRAINTES COUCHES INTERNES **************
# *********************************************************
def contrainte_ReLU_Mix(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        num_contrainte : int, 
        par_couches : bool = False,
        neurones_actifs_stables : List = [],
        neurones_inactifs_stables : List = []):
    """A combiner avec la contrainte sur les bornes des zk pour les couches internes (avec zk >= 0)"""
    # ***** Contrainte :  zk+1 >= Wk zk + bk *****
    # ***** Nombre de contraintes : 2 * sum(n[1:K]) *****************
    for k in range(1, K):
        for j in range(n[k]):
            # Traitement des neurones stables *******************************
            if (k, j) in neurones_actifs_stables:
                if par_couches :
                    A2_k = [1+n[k-1]+j] + [(1+i) for i in range(n[k - 1])]
                    A2_l = [0] * (len(A2_k))
                    A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                    task.putbarablocktriplet( [num_contrainte] * len(A2_k), [k-1] * len(A2_k),
                    A2_k, A2_l, A2_v)
                else : 
                    idx_k_prec = return_i_from_k_j__variable_z(k - 1, 0, n)
                    idx_k_j = return_i_from_k_j__variable_z(k, j, n)
                    A2_k = [idx_k_j] + [(idx_k_prec+i) for i in range(n[k - 1])]
                    A2_l = [0] * (len(A2_k))
                    A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                    task.putbarablocktriplet( [num_contrainte] * len(A2_k), [0] * len(A2_k),
                        A2_k, A2_l, A2_v)
                task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [b[k - 1][j]], [b[k - 1][j]])
                num_contrainte += 1
                continue
            elif (k,j) in neurones_inactifs_stables:
                if par_couches : 
                    task.putbarablocktriplet( [num_contrainte], [k-1],
                    [1 + n[k - 1] + j], [0], [1])
                else :
                    task.putbarablocktriplet( [num_contrainte], [0],
                    [return_i_from_k_j__variable_z(k, j, n)], [0], [1])
                task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
                num_contrainte += 1
                continue

            # ***************** Traitement des neurones instables *******************************
            if par_couches :
                    A2_k = [1+n[k-1]+j] + [(1+i) for i in range(n[k - 1])]
                    A2_l = [0] * (len(A2_k))
                    A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                    task.putbarablocktriplet( [num_contrainte] * len(A2_k), [k-1] * len(A2_k),
                    A2_k, A2_l, A2_v)
            else : 
                idx_k_prec = return_i_from_k_j__variable_z(k - 1, 0, n)
                idx_k_j = return_i_from_k_j__variable_z(k, j, n)
                A2_k = [idx_k_j] + [(idx_k_prec+i) for i in range(n[k - 1])]
                A2_l = [0] * (len(A2_k))
                A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                task.putbarablocktriplet( [num_contrainte] * len(A2_k), [0] * len(A2_k),
                    A2_k, A2_l, A2_v)
            task.putconboundlist([num_contrainte], [mosek.boundkey.lo], [b[k - 1][j]], [inf])
            num_contrainte += 1
            continue

    # ***** Contrainte :  zk+1 x (zk+1 - Wk zk - bk)  == 0  *****
    for k in range(1, K):
        for j in range(n[k]):
            # La contrainte des neurones stables a déjà été traitée *******************************
            if (k,j) in neurones_actifs_stables+neurones_inactifs_stables:
                continue
            # ************  Traitement des neurones instables *******************************
            if par_couches:
                A3_k = (
                    [1 + n[k - 1] + j]
                    + [1 + n[k - 1] + j]
                    + [1 + n[k - 1] + j] * (n[k - 1])
                )
                A3_l = [0] + [1 + n[k - 1] + j] + [(1 + i) for i in range(n[k - 1])]
                A3_v = (
                    [-b[k - 1][j] / 2]
                    + [1]
                    + [-W[k - 1][j][i] / 2 for i in range(n[k - 1])]
                )

                task.putbarablocktriplet(
                    [num_contrainte] * (len(A3_k)),
                    [k-1] * (len(A3_k)),
                    A3_k,
                    A3_l,
                    A3_v,
                )
            else:
                idx = return_i_from_k_j__variable_z(k - 1, 0, n)

                A3_k = (
                    [return_i_from_k_j__variable_z(k, j, n)]
                    + [return_i_from_k_j__variable_z(k, j, n)]
                    + [return_i_from_k_j__variable_z(k, j, n)] * (n[k - 1])
                )
                A3_l = (
                    [0]
                    + [return_i_from_k_j__variable_z(k, j, n)]
                    + [(idx + i) for i in range(n[k - 1])]
                )
                A3_v = (
                    [-b[k - 1][j] / 2]
                    + [1]
                    + [-W[k - 1][j][i] / 2 for i in range(n[k - 1])]
                )

                task.putbarablocktriplet(
                    [num_contrainte] * (len(A3_k)),
                    [0] * (len(A3_k)),
                    A3_k,
                    A3_l,
                    A3_v,
                )
            task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
            num_contrainte += 1
    return num_contrainte


def contrainte_ReLU_Glover(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        W : List[List[List[float]]], 
        b : List[List[float]], 
        U : List[List[float]], 
        L : List[List[float]], 
        num_contrainte : int,
        par_couches : bool = False,
        neurones_actifs_stables  : List = [],
        neurones_inactifs_stables : List = []):
    # ***** Contrainte : l (1-sigma) <= Wz+b <= u sigma *****
    for k in range(1, K):
        for j in range(n[k]):
            if (k, j) in neurones_actifs_stables:
                if par_couches :
                    A2_k = [1+n[k-1]+j] + [(1+i) for i in range(n[k - 1])]
                    A2_l = [0] * (len(A2_k))
                    A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                    task.putbarablocktriplet( [num_contrainte] * len(A2_k), [k-1] * len(A2_k),
                    A2_k, A2_l, A2_v)
                else : 
                    idx_k_prec = return_i_from_k_j__variable_z(k - 1, 0, n)
                    idx_k_j = return_i_from_k_j__variable_z(k, j, n)
                    A2_k = [idx_k_j] + [(idx_k_prec+i) for i in range(n[k - 1])]
                    A2_l = [0] * (len(A2_k))
                    A2_v = [1/2] + [-(W[k-1][j][i])/2 for i in range(n[k - 1])]
                    task.putbarablocktriplet( [num_contrainte] * len(A2_k), [0] * len(A2_k),
                        A2_k, A2_l, A2_v)
                task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [b[k - 1][j]], [b[k - 1][j]])
                num_contrainte += 1
                continue

            idx = return_i_from_k_j__variable_z(k - 1, 0, n)
            # Partie z
            A1_z_k = [(idx + i) for i in range(n[k - 1])]
            A1_z_l = [0] * (n[k - 1])
            A1_z_v = [W[k - 1][j][i] / 2 for i in range(n[k - 1])]


            # Contrainte : Wz - u sigma <= -b
            # Partie sigma
            idx_sigma = return_i_from_k_j__variable_sigma(k, j, n)
            A1_sigma_k = [idx_sigma]
            A1_sigma_l = [0]
            A1_sigma_v = [-U[k][j] / 2]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A1_sigma_k) + len(A1_z_k)),
                [0] * (len(A1_sigma_k) + len(A1_z_k)),
                A1_sigma_k + A1_z_k,
                A1_sigma_l + A1_z_l,
                A1_sigma_v + A1_z_v,
            )

            task.putconboundlist(
                [num_contrainte],
                [mosek.boundkey.up],
                [-inf],
                [-b[k - 1][j]],
            )
            num_contrainte += 1

            # Contrainte : Wz + l sigma <= l - b
            # Partie sigma
            idx_sigma = return_i_from_k_j__variable_sigma(k, j, n)
            A1_sigma_k = [idx_sigma]
            A1_sigma_l = [0]
            A1_sigma_v = [L[k][j] / 2]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A1_sigma_k) + len(A1_z_k)),
                [0] * (len(A1_sigma_k) + len(A1_z_k)),
                A1_sigma_k + A1_z_k,
                A1_sigma_l + A1_z_l,
                A1_sigma_v + A1_z_v,
            )

            task.putconboundlist(
                [num_contrainte],
                [mosek.boundkey.lo],
                [-b[k - 1][j] + L[k][j]],
                [inf],
            )
            num_contrainte += 1

    # ***** Contrainte : sigmaK x (WK zK + bK) = zK+1 *****
    for k in range(1, K):
        for j in range(n[k]):
            if (k, j) in neurones_actifs_stables + neurones_inactifs_stables:
                continue
            idx = return_i_from_k_j__variable_z(k - 1, 0, n)
            idx_sigma = return_i_from_k_j__variable_sigma(k, j, n)
            # Partie sigma * z
            A3_zsigma_k = [idx_sigma] * (n[k - 1])
            A3_zsigma_l = [(idx + i) for i in range(n[k - 1])]
            A3_zsigma_v = [W[k - 1][j][i] / 2 for i in range(n[k - 1])]

            # Partie z
            A3_z_k = [return_i_from_k_j__variable_z(k, j, n)]
            A3_z_l = [0]
            A3_z_v = [-1 / 2]

            # Partie sigma
            A3_sigma_k = [1 + sum(n) + (sum(n[1:k])) + j]
            A3_sigma_l = [0]
            A3_sigma_v = [b[k - 1][j] / 2]
            task.putbarablocktriplet(
                [num_contrainte]
                * (
                    len(A3_sigma_k) + len(A3_z_k) + len(A3_zsigma_k)
                ),  # Numero de la contrainte
                [0]
                * (
                    len(A3_sigma_k) + len(A3_z_k) + len(A3_zsigma_k)
                ),  # Variable SDP sigma (numero 2)
                A3_sigma_k + A3_z_k + A3_zsigma_k,  # Entries: (k,l)->v
                A3_sigma_l + A3_z_l + A3_zsigma_l,
                A3_sigma_v + A3_z_v + A3_zsigma_v,
            )

            task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
            num_contrainte += 1
    return num_contrainte


def contrainte_sigma_borne(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        num_contrainte : int):
    # Contrainte :  0 <= sigma_(k,j) <= 1 *****************
    # ***** Nombre de contraintes : sum(n[1:K]) *****************
    for k in range(1, K):
        for j in range(n[k]):
            idx_sigma = return_i_from_k_j__variable_sigma(k, j, n)
            task.putbarablocktriplet(
                [num_contrainte],
                [0],  #  Variable z sigma
                [idx_sigma],
                [0],
                [0.5],
            )
            # Bornes
            task.putconboundlist([num_contrainte], [mosek.boundkey.ra], [0], [1])
            num_contrainte += 1
    return num_contrainte



def contrainte_sigma_discret(
        task : mosek.Task, 
        K : int, 
        n : List[int], 
        num_contrainte : int):
    # Contrainte : sigma_(k,j) ^2 == sigma_(k,j) ***************
    # ***** Nombre de contraintes : sum(n[1:K]) *****************
    for k in range(1, K):
        for j in range(n[k]):
            idx_sigma = return_i_from_k_j__variable_sigma(k, j, n)
            A10_k = [idx_sigma, idx_sigma]
            A10_l = [0, idx_sigma]
            A10_v = [1 / 2, -1]
            task.putbarablocktriplet(
                [num_contrainte] * (len(A10_k)),
                [0] * (len(A10_k)),  #  Variable z sigma
                A10_k,
                A10_l,
                A10_v,
            )
            # Bornes
            task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [0], [0])
            num_contrainte += 1

    return num_contrainte
