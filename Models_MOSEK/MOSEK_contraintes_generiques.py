import numpy as np
import mosek
from typing import List, Tuple
from MOSEK_outils import return_good_j_beta, return_i_from_k_j__variable_z, return_k_j_from_i
inf = 10e5


# *********** Contrainte de borne sur la boule initiale *************
def contrainte_boule_initiale(
        task : mosek.Task,
        n : int,
        x0 : List[float],
        epsilon : float,
        U : List[List[float]],
        L : List[List[float]],
        num_contrainte : int):
    # Contrainte :  x0 - epsilon <= z0 <= x0 + epsilon
    # ***** Nombre de contraintes : n[0] *****************
    for j in range(n[0]):
        A9_k = [j + 1]
        A9_l = [0]
        A9_v = [1 / 2]
        task.putbarablocktriplet(
            [num_contrainte] * (len(A9_k)),  
            [0] * (len(A9_k)),  
            A9_k,  
            A9_l,
            A9_v,
        )
        # Bornes
        task.putconboundlist(
            [num_contrainte],
            [mosek.boundkey.ra],
            [max(x0[j] - epsilon,L[0][j])],
            [min(x0[j] + epsilon,U[0][j])],
        )
        num_contrainte += 1
    return num_contrainte


# ********* Contrainte imposant que le premier terme de la matrice soit égal à 1 *****
def contrainte_premier_terme_egal_a_1(
        task : mosek.Task,
        K : int,
        num_variables : int,
        num_contrainte : int):  
    # ***** Nombre de contraintes : num_variables *****************
    for var in range(num_variables):
        task.putbarablocktriplet(
            [num_contrainte],
            [var],
            [0],
            [0],
            [1],
        )
        # Bornes
        task.putconboundlist([num_contrainte], [mosek.boundkey.fx], [1], [1])
        num_contrainte += 1
    return num_contrainte