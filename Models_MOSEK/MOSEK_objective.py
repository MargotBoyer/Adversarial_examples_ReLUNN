import numpy as np
import mosek
from MOSEK_outils import return_i_from_k_j__variable_z,return_good_j_beta
from typing import List, Tuple
inf = 10e5


def objective_function_diff(
        task : mosek.Task,
        K: int,
        n : List[int],
        ytrue : int,
        numvar : int):
    # Fonction objectif : zK_ytrue - somme(zK_y pour tous y != ytrue)
    for i in range(numvar):
        task.putcj(i, 0)
    # Partie SDP de la fonction objectif
    idx_last_layer = return_i_from_k_j__variable_z(K, 0, n)
    task.putbarcblocktriplet(
        [0] * n[K],  
        [idx_last_layer + ytrue]
        + [
            (idx_last_layer + i) for i in range(n[K]) if i != ytrue
        ], 
        [0] * n[K],
        [1 / 2] + [-1 / 2] * (n[K] - 1),
            )
    

def objective_function_diff_ycible(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        ycible : int,
        W : List[List[List[float]]],
        numvar : int,
        par_couches : bool = False,
        derniere_couche_lineaire : bool = True):
    # Fonction objectif : zK_ytrue - somme(zK_y pour tous y != ytrue)
    for i in range(numvar):
        task.putcj(i, 0)
    # Partie SDP de la fonction objectif

    if par_couches and derniere_couche_lineaire : 
        task.putbarcblocktriplet(
            [K-1] * 2,  
            [1 + n[K-1] + ytrue, 1 + n[K-1] + ycible], 
            [0] * 2,
            [1 / 2, -1 / 2] )
    elif not par_couches and derniere_couche_lineaire : 
        idx_last_layer = return_i_from_k_j__variable_z(K, 0, n)
        task.putbarcblocktriplet(
            [0] * 2,  
            [idx_last_layer + ytrue, idx_last_layer + ycible], 
            [0] * 2,
            [1 / 2, -1 / 2] )
    elif par_couches and not derniere_couche_lineaire :
        task.putbarcblocktriplet(
            [K-2] * n[K-2],  
            [(1 + n[K-2] + i) for i in range(n[K-1])], 
            [0] * n[K-1],
            [(W[K-1][ytrue][i] - W[K-1][ycible][i])/2 for i in range(n[K-1])] )
    elif not par_couches and not derniere_couche_lineaire :
        idx_last_layer = return_i_from_k_j__variable_z(K-1, 0, n)
        task.putbarcblocktriplet(
            [0] * n[K-1],  
            [(idx_last_layer + i) for i in range(n[K-1])], 
            [0] * n[K-1],
            [(W[K-1][ytrue][i] - W[K-1][ycible][i])/2 for i in range(n[K-1])] )
    


def objective_function_diff_betas(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        numvar : int,
        sigmas : bool = False, 
        par_couches : bool = False):
    # Fonction objectif : zK_ytrue - somme(zK_y * Beta_y pour tous y != ytrue)
    for i in range(numvar):
        task.putcj(i, 0)
    # Partie SDP de la fonction objectif
    
    if par_couches : 
        task.putbarcblocktriplet(
            [K-1] * n[K],  
            [1 + n[K-1] + ytrue] + [(n[K-1] + n[K] + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue],
            [0]+ [(1 + n[K-1] + i) for i in range(n[K]) if i != ytrue], 
            [1 / 2] + [-1 / 2] * (n[K] - 1),
                )

    else : 
        idx_last_layer = return_i_from_k_j__variable_z(K, 0, n)
        if sigmas : 
            task.putbarcblocktriplet(
                [0] * n[K],  
                [idx_last_layer + ytrue] + [(sum(n) + sum(n[1:K]) + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue],
                [0]+ [(idx_last_layer + i) for i in range(n[K]) if i != ytrue], 
                [1 / 2] + [-1 / 2] * (n[K] - 1),
                    )
        else : 
            task.putbarcblocktriplet(
                [0] * n[K],  
                [1 + sum(n[0:K]) + ytrue] + [(sum(n) + return_good_j_beta(i, ytrue)) for i in range(n[K]) if i != ytrue],
                [0]+ [(idx_last_layer + i) for i in range(n[K]) if i != ytrue], 
                [1 / 2] + [-1 / 2] * (n[K] - 1),
                    )


def objective_function_diff_par_couches(
        task : mosek.Task,
        K : int,
        n : List[int],
        ytrue : int,
        numvar : int):
    # Fonction objectif :  zK_ytrue - somme(zK_y pour tous y != ytrue)
    for i in range(numvar):
        task.putcj(i, 0)
    # Partie SDP de la fonction objectif
    task.putbarcblocktriplet(
        [K - 1] * n[K],  
        [1 + n[K-1] +ytrue]
        + [
            (1 + n[K-1] + i) for i in range(n[K]) if i != ytrue
        ], 
        [0] *n[K],
        [1 / 2] + [-1 / 2] * (n[K] - 1),
        )